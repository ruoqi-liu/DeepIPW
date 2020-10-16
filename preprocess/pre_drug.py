import argparse
from collections import defaultdict
import pickle
import numpy as np
from datetime import datetime
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='process parameters')
    parser.add_argument('--input_data_dir', default='../data/CAD/drug', help='input data directory')
    parser.add_argument('--output_data_dir', default='pickles', help='output data directory')

    args = parser.parse_args()
    return args


def ndc2rxing():
    mapping = np.loadtxt(fname='../data/NDC_complete_mapping.csv', delimiter=',', dtype='str', skiprows=1,
                         usecols=(1, 2))
    ndc2rx_mapping = {ndc: rx for (rx, ndc) in mapping}

    return ndc2rx_mapping


def pre_drug_table(args):
    # data format: {drug_id:
    #                   {{patient_id:
    #                       ((take_date, take_days), (take_date, take_days),...)},
    #                   {patient_id:
    #                       ((take_date, take_days), (take_date, take_days),...)},...},
    #               {drug_id:
    #                   {{patient_id:
    #                       ((take_date, take_days), (take_date, take_days),...)},
    #                   {patient_id:
    #                       ((take_date, take_days), (take_date, take_days),...)},...},...}
    cad_prescription_taken_by_patient = defaultdict(dict)
    ndc2rx_mapping = ndc2rxing()

    for dir in tqdm(args.input_data_dir):
        for year in tqdm(range(12, 18)):
            print('dir: {}\tyear: {}'.format(dir, year), flush=True)
            file = '{}/drug{}.csv'.format(dir, year)
            with open(file, 'r') as f:
                next(f)
                for row in f:
                    row = row.strip('\n')
                    row = row.split(',')
                    enroll_id, prescription, date, day = row[0], row[1], row[2], row[3]
                    if date and day:
                        if prescription in ndc2rx_mapping:
                            prescription_rx = ndc2rx_mapping.get(prescription)
                            if prescription_rx not in cad_prescription_taken_by_patient:
                                cad_prescription_taken_by_patient[prescription_rx][enroll_id] = set([(date, day)])
                            else:
                                if enroll_id not in cad_prescription_taken_by_patient.get(prescription_rx):
                                    cad_prescription_taken_by_patient[prescription_rx][enroll_id] = set([(date, day)])
                                else:
                                    cad_prescription_taken_by_patient[prescription_rx][enroll_id].add((date, day))

    try:
        print('dumping...', flush=True)
        out = '{}/cad_prescription_taken_by_patient.pkl'.format(args.output_data_dir)
        pickle.dump(cad_prescription_taken_by_patient,
                    open(out, 'wb'))

    except Exception as e:
        print(e)

    print('finish dump', flush=True)

    return cad_prescription_taken_by_patient


def concomitant_drugs_extractor(cad_prescription_taken_by_patient, n_patient, n_prescription, time_interval):

    user_cohort = defaultdict(list)
    print('number of drugs: {}'.format(len(cad_prescription_taken_by_patient)), flush=True)
    for drug in cad_prescription_taken_by_patient.keys():
        patient_take_times = cad_prescription_taken_by_patient.get(drug)

        if len(patient_take_times) < n_patient:
            continue

        count = 0
        for patient, take_times in patient_take_times.items():
            if drug_time_interval_is_valid(take_times, n_prescription, time_interval):
                count += 1
            if count >= n_patient:
                user_cohort[drug].append(patient)

    print('number of concomitant drugs: {}'.format(len(user_cohort.keys())), flush=True)
    pickle.dump(user_cohort, open('../res/8.7/user_cohort.pkl', 'wb'))



def drug_time_interval_is_valid(take_times, n_prescription, time_interval):
    count = 0
    dates = [datetime.strptime(pair[0], '%m/%d/%Y') for pair in take_times if pair[0] and pair[1]]
    dates = sorted(dates)
    for i in range(1, len(dates)):
        if (dates[i] - dates[i - 1]).days >= time_interval:
            count += 1
        if count >= n_prescription:
            return True
    return False


def my_dump(obj, file_name):
    print('dumping...')
    pickle.dump(obj, open(file_name, 'wb'))


if __name__ == '__main__':
    pre_drug_table(args=parse_args())



