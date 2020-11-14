import argparse
from collections import defaultdict
import pickle
import numpy as np
from datetime import datetime
import os


def parse_args():
    parser = argparse.ArgumentParser(description='process parameters')
    parser.add_argument('--input_data_dir', default='../data/synthetic/drug', help='input data directory')
    parser.add_argument('--output_data_dir', default='pickles/cad_prescription_taken_by_patient.pkl', help='output data directory')

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

    files = os.listdir(args.input_data_dir)
    for file in files:
        print('dir: {}\tfile: {}'.format(args.input_data_dir, file), flush=True)
        df = os.path.join(args.input_data_dir, file)
        with open(df, 'r') as f:
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
        pickle.dump(cad_prescription_taken_by_patient,
                    open(args.output_data_dir, 'wb'))

    except Exception as e:
        print(e)

    print('finish dump', flush=True)

    print('# of Drugs: {}\t'.format(len(cad_prescription_taken_by_patient)))

    return cad_prescription_taken_by_patient


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


if __name__ == '__main__':
    pre_drug_table(args=parse_args())



