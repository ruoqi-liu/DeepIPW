from collections import defaultdict
from datetime import  datetime
import pickle
from tqdm import tqdm

def my_dump(obj, file_name):
    print('dumping...', flush=True)
    pickle.dump(obj, open(file_name, 'wb'))
    print('dumped...', flush=True)


def my_load(file_name):

    print('loading...', flush=True)
    return pickle.load(open(file_name, 'rb'))


def is_valid_outcome_range(dx, code_range):
    for code in code_range:
        if dx.startswith(code):
            return True
    return False

def is_valid_outcome_code(dx, codes):
    return dx in codes

def pre_user_cohort_outcome(patient_list, codes9, codes0):
    cad_user_cohort_outcome = defaultdict(list)
    for dir in ['CAD2012/', 'CAD2013-2016/']:
        print('dir: {}'.format(dir), flush=True)
        for i in range(12, 15):
            print('year: {}'.format(i), flush=True)
            print('inpatient', flush=True)
            patient_dx_file = '../data/' + dir + 'inpatient/inpat' + str(i) + '.csv'
            with open(patient_dx_file, 'r') as f:
                next(f)
                for row in f:
                    row = row.split(',')
                    enrolid, dxs, date = row[0], row[1:16], row[-2]
                    dxs = [dx for dx in dxs if dx]
                    if date and enrolid in patient_list:
                        for dx in dxs:
                            if is_valid_outcome_range(dx, codes9):
                                cad_user_cohort_outcome[enrolid].append(date)
                            break

            print('outpatient', flush=True)
            patient_dx_file = '../data/' + dir + 'outpatient/outpat' + str(i) + '.csv'
            with open(patient_dx_file, 'r') as f:
                next(f)
                for row in f:
                    row = row.split(',')
                    enrolid, dxs, date = row[0], row[1:5], row[-1]
                    dxs = [dx for dx in dxs if dx]
                    if date and enrolid in patient_list:
                        for dx in dxs:
                            if is_valid_outcome_range(dx, codes9):
                                cad_user_cohort_outcome[enrolid].append(date)
                            break

        for i in range(15, 18):
            print('year: {}'.format(i), flush=True)
            print('inpatient', flush=True)
            patient_dx_file = '../data/' + dir + 'inpatient/inpat' + str(i) + '.csv'
            with open(patient_dx_file, 'r') as f:
                next(f)
                for row in f:
                    row = row.split(',')
                    enrolid, dxs, type, date = row[0], row[1:16], row[16], row[-2]
                    dxs = [dx for dx in dxs if dx]
                    if date and enrolid in patient_list:
                        if type == '9':
                            for dx in dxs:
                                if is_valid_outcome_range(dx, codes9):
                                    cad_user_cohort_outcome[enrolid].append(date)
                                break
                        else:
                            for dx in dxs:
                                if is_valid_outcome_range(dx, codes0):
                                    cad_user_cohort_outcome[enrolid].append(date)
                                break

            print('outpatient', flush=True)
            patient_dx_file = '../data/' + dir + 'outpatient/outpat' + str(i) + '.csv'
            with open(patient_dx_file, 'r') as f:
                next(f)
                for row in f:
                    row = row.split(',')
                    enrolid, dxs, type, date = row[0], row[1:5], row[6], row[-1]
                    dxs = [dx for dx in dxs if dx]
                    if date and enrolid in patient_list:
                        if type == '9':
                            for dx in dxs:
                                if is_valid_outcome_range(dx, codes9):
                                    cad_user_cohort_outcome[enrolid].append(date)
                                break
                        else:
                            for dx in dxs:
                                if is_valid_outcome_range(dx, codes0):
                                    cad_user_cohort_outcome[enrolid].append(date)
                                break


    # my_dump(cad_user_cohort_outcome, dump_file)

    return cad_user_cohort_outcome

def get_patient_list(min_patient, cad_prescription_taken_by_patient):
    patients_list = set()
    for drug, patients in cad_prescription_taken_by_patient.items():
        if len(patients) >= min_patient:
            for patient in patients:
                patients_list.add(patient)
    return patients_list

# if __name__ == '__main__':
#
#
#     codes9 = ['425', '428', '40201','40211', '40291','40401','40403', '40411', '40413','40491','40493','K77']
#     codes0 = ['I11', 'I13', 'I50', 'I42', 'K77']

    # Stroke
    # codes9 = ['4380', '4381', '4382', '4383', '4384', '4385', '4386', '4387', '4388', '4389', 'V1254']
    # codes0 = ['Z8673', 'I60', 'I61', 'I62', 'I63', 'I64', 'I65', 'I66', 'I67', 'I68', 'I69', 'G458', 'G459']

    # cad_prescription_taken_by_patient = my_load('../res/8.7/cad_prescription_taken_by_patient_exclude_30.pkl')
    # min_patient = 500
    # patient_list = get_patient_list(min_patient, cad_prescription_taken_by_patient)
    #
    # dump_file = '../res/01.28/user_cohort_outcome_30_hf.pkl'
    # pre_user_cohort_outcome(patient_list, codes9, codes0, dump_file)

    # outcome = pickle.load(open('../res/01.28/user_cohort_outcome_30_hf.pkl', 'rb'))
    # print()