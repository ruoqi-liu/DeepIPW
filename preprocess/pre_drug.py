from collections import defaultdict
import pickle
import numpy as np
from datetime import datetime

def ndc2rxing():
    mapping = np.loadtxt(fname='../data/NDC_complete_mapping.csv', delimiter=',', dtype='str', skiprows=1, usecols=(1,2))
    ndc2rx_mapping = {ndc: rx for (rx, ndc) in mapping}

    return ndc2rx_mapping



def pre_drug_table(outfile):

    cad_prescription_taken_by_patient = defaultdict(dict)
    cad_patient_take_prescription = defaultdict(dict)

    ndc2rx_mapping = ndc2rxing()

    for dir in ['2012','2013-2016']:
        for year in range(12, 18):
            print('dir: {}\tyear: {}'.format(dir, year), flush=True)
            file = '../data/CAD' + dir + '/drug/drug' + str(year) + '.csv'
            with open(file, 'r') as f:
                next(f)
                for row in f:
                    row = row.strip('\n')
                    row = row.split(',')
                    enroll_id, prescription, date, day = row[0], row[1], row[2], row[3]
                    if date and day:
                        if prescription in ndc2rx_mapping:
                            prescription_rx = ndc2rx_mapping.get(prescription)
                            if enroll_id not in cad_patient_take_prescription:
                                cad_patient_take_prescription[enroll_id][prescription_rx] = set([(date, day)])
                            else:
                                if prescription_rx not in cad_patient_take_prescription.get(enroll_id):
                                    cad_patient_take_prescription[enroll_id][prescription_rx] = set([(date, day)])
                                else:
                                    cad_patient_take_prescription[enroll_id][prescription_rx].add((date, day))

                            if prescription_rx not in cad_prescription_taken_by_patient:
                                cad_prescription_taken_by_patient[prescription_rx][enroll_id] = set([(date, day)])
                            else:
                                if enroll_id not in cad_prescription_taken_by_patient.get(prescription_rx):
                                    cad_prescription_taken_by_patient[prescription_rx][enroll_id] = set([(date, day)])
                                else:
                                    cad_prescription_taken_by_patient[prescription_rx][enroll_id].add((date, day))


    try:
        print('dumping...', flush=True)
        out1 = outfile + 'cad_prescription_taken_by_patient.pkl'
        pickle.dump(cad_prescription_taken_by_patient,
                    open(out1, 'wb'))

        out2 = outfile + 'cad_patient_take_prescription.pkl'
        pickle.dump(cad_patient_take_prescription,
                    open(out2, 'wb'))
    except Exception as e:
        print(e)

    print('finish dump', flush=True)

    return cad_prescription_taken_by_patient


def concomitant_drugs_extractor(cad_prescription_taken_by_patient, n_patient, n_prescription, time_interval):

    # concomitant_drugs = set()
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

    # pickle.dump(concomitant_drugs, open('../pickles/concomitant_drugs.pkl', 'wb'))


def drug_time_interval_is_valid(take_times, n_prescription, time_interval):
    count = 0
    dates = [datetime.strptime(pair[0], '%m/%d/%Y') for pair in take_times if pair[0] and pair[1]]
    dates = sorted(dates)
    for i in range(1, len(dates)):
        if (dates[i] - dates[i-1]).days >= time_interval:
            count += 1
        if count >= n_prescription:
            return True
    return False


def my_dump(obj, file_name):

    print('dumping...')
    pickle.dump(obj, open(file_name, 'wb'))




if __name__ == '__main__':
    cad_prescription_taken_by_patient = pickle.load(open('../pickles/cad_prescription_taken_by_patient.pkl', 'rb'))
    # concomitant_drugs_extractor(cad_prescription_taken_by_patient, 500, 2, 30)
    print('n_drugs: {}'.format(len(cad_prescription_taken_by_patient)))

   

