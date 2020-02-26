from collections import defaultdict
from datetime import datetime
import pickle
from tqdm import tqdm

def my_dump(obj, file_name):
    print('dumping...', flush=True)
    pickle.dump(obj, open(file_name, 'wb'))
    print('dumped...', flush=True)


def my_load(file_name):

    print('loading...', flush=True)
    return pickle.load(open(file_name, 'rb'))



def get_user_dx(patient_list, icd9_to_css, icd10_to_css):

    user_dx = defaultdict(dict)
    for dir in ['CAD2012/', 'CAD2013-2016/']:
        print('dir: {}'.format(dir), flush= True)
        for i in range(12, 15):
            print('year: {}'.format(i), flush=True)
            print('inpatient', flush=True)
            patient_dx_file = '../data/' + dir + 'inpatient/inpat' + str(i) + '.csv'
            with open(patient_dx_file, 'r') as f:
                next(f)
                for row in f:
                    row = row.strip('\n')
                    row = row.split(',')
                    enrolid, dxs, date = row[0], row[1:16], row[-2]
                    dxs = get_css_code_for_icd(dxs, icd9_to_css)
                    if dxs and enrolid in patient_list:
                        if enrolid not in user_dx:
                            user_dx[enrolid][date] = dxs
                        else:
                            if date not in user_dx[enrolid]:
                                user_dx[enrolid][date] = dxs
                            else:
                                user_dx[enrolid][date].extend(dxs)

            print('outpatient', flush=True)
            patient_dx_file = '../data/' + dir + 'outpatient/outpat' + str(i) + '.csv'
            with open(patient_dx_file, 'r') as f:
                next(f)
                for row in f:
                    row = row.strip('\n')
                    row = row.split(',')
                    enrolid, dxs, date = row[0], row[1:5], row[-1]
                    dxs = get_css_code_for_icd(dxs, icd9_to_css)
                    if dxs and enrolid in patient_list:
                        if enrolid not in user_dx:
                            user_dx[enrolid][date] = dxs
                        else:
                            if date not in user_dx[enrolid]:
                                user_dx[enrolid][date] = dxs
                            else:
                                user_dx[enrolid][date].extend(dxs)



        for i in range(15, 18):
            print('year: {}'.format(i), flush=True)
            print('inpatient', flush=True)
            patient_dx_file = '../data/' + dir + 'inpatient/inpat' + str(i) + '.csv'
            with open(patient_dx_file, 'r') as f:
                next(f)
                for row in f:
                    row = row.strip('\n')
                    row = row.split(',')
                    enrolid, dxs, type, date = row[0], row[1:16], row[16], row[-2]
                    if date and enrolid in patient_list:
                        if type == '9':
                            dxs = get_css_code_for_icd(dxs, icd9_to_css)
                            if dxs:
                                if enrolid not in user_dx:
                                    user_dx[enrolid][date] = dxs
                                else:
                                    if date not in user_dx[enrolid]:
                                        user_dx[enrolid][date] = dxs
                                    else:
                                        user_dx[enrolid][date].extend(dxs)
                        elif type == '0':
                            dxs = get_css_code_for_icd(dxs, icd10_to_css)
                            if dxs:
                                if enrolid not in user_dx:
                                    user_dx[enrolid][date] = dxs
                                else:
                                    if date not in user_dx[enrolid]:
                                        user_dx[enrolid][date] = dxs
                                    else:
                                        user_dx[enrolid][date].extend(dxs)

            print('outpatient', flush=True)
            patient_dx_file = '../data/' + dir + 'outpatient/outpat' + str(i) + '.csv'
            with open(patient_dx_file, 'r') as f:
                next(f)
                for row in f:
                    row = row.strip('\n')
                    row = row.split(',')
                    enrolid, dxs, type, date = row[0], row[1:5], row[5], row[-1]
                    if date and enrolid in patient_list:
                        if type == '9':
                            dxs = get_css_code_for_icd(dxs, icd9_to_css)
                            if dxs:
                                if enrolid not in user_dx:
                                    user_dx[enrolid][date] = dxs
                                else:
                                    if date not in user_dx[enrolid]:
                                        user_dx[enrolid][date] = dxs
                                    else:
                                        user_dx[enrolid][date].extend(dxs)
                        elif type == '0':
                            dxs = get_css_code_for_icd(dxs, icd10_to_css)
                            if dxs:
                                if enrolid not in user_dx:
                                    user_dx[enrolid][date] = dxs
                                else:
                                    if date not in user_dx[enrolid]:
                                        user_dx[enrolid][date] = dxs
                                    else:
                                        user_dx[enrolid][date].extend(dxs)


    # my_dump(user_dx, dump_file)
    return user_dx


def get_css_code_for_icd(icd_codes, icd_to_css):
    css_codes = []
    for icd_code in icd_codes:
        for i in range(len(icd_code), -1, -1):
            if icd_code[:i] in icd_to_css:
                css_codes.append(icd_to_css.get(icd_code[:i]))

    return css_codes



    

def pre_user_cohort_dx(user_dx, cad_prescription_taken_by_patient,min_patients):
    user_cohort_dx = AutoVivification()
    for drug, taken_by_patient in tqdm(cad_prescription_taken_by_patient.items()):
        if len(taken_by_patient.keys()) >= min_patients:
            for patient, taken_times in taken_by_patient.items():
                index_date = taken_times[0]
                date_codes = user_dx.get(patient)
                for date, codes in date_codes.items():
                    date = datetime.strptime(date, '%m/%d/%Y')
                    if date < index_date:
                        # if drug not in user_cohort_dx:
                        #     user_cohort_dx[drug][patient] = [(codes, date)]
                        # else:
                        #     if patient not in user_cohort_dx[drug]:
                        #         user_cohort_dx[drug][patient] = [(codes, date)]
                        #     else:
                        #         user_cohort_dx[drug][patient].append((codes, date))
                        if drug not in user_cohort_dx:
                            user_cohort_dx[drug][patient][date] = set(codes)
                        else:
                            if patient not in user_cohort_dx[drug]:
                                user_cohort_dx[drug][patient][date] = set(codes)
                            else:
                                if date not in user_cohort_dx[drug][patient]:
                                    user_cohort_dx[drug][patient][date] = set(codes)
                                else:
                                    user_cohort_dx[drug][patient][date].union(codes)

    # my_dump(user_cohort_dx, dump_file)

    return user_cohort_dx


class AutoVivification(dict):
    """Implementation of perl's autovivification feature."""
    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value


def get_patient_list(min_patient, cad_prescription_taken_by_patient):
    patients_list = set()
    for drug, patients in cad_prescription_taken_by_patient.items():
        if len(patients) >= min_patient:
            for patient in patients:
                patients_list.add(patient)
    return patients_list


def get_user_cohort_dx(cad_prescription_taken_by_patient, icd9_to_css, icd10_to_css, min_patient):
    patient_list = get_patient_list(min_patient, cad_prescription_taken_by_patient)
    user_dx = get_user_dx(patient_list, icd9_to_css, icd10_to_css)
    return pre_user_cohort_dx(user_dx, cad_prescription_taken_by_patient, min_patient)


if __name__ == '__main__':

    # cad_prescription_taken_by_patient = my_load('../res/8.7/cad_prescription_taken_by_patient_exclude_30.pkl')
    # min_patient = 500
    # icd9_to_css = my_load('../pickles/icd9_to_css.pkl')
    # icd10_to_css = my_load('../pickles/icd10_to_css.pkl')
    # dump_file = '../res/10.15/user_cohort_dx_30.pkl'
    #
    # get_user_cohort_dx(cad_prescription_taken_by_patient, icd9_to_css, icd10_to_css, min_patient,
    #                    dump_file)

    user_cohort_dx = my_load('../res/10.15/user_cohort_dx_30.pkl')
    print()