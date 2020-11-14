from collections import defaultdict
from datetime import datetime
import os
import pandas as pd
from tqdm import tqdm



def get_user_dx(indir, patient_list, icd9_to_css, icd10_to_css):

    user_dx = defaultdict(dict)

    inpatient_dir = os.path.join(indir, 'inpatient')
    inpatient_files = os.listdir(inpatient_dir)
    outpatient_dir = os.path.join(indir, 'outpatient')
    outpatient_files = os.listdir(outpatient_dir)

    files = [os.path.join(inpatient_dir, file) for file in inpatient_files] \
            + [os.path.join(outpatient_dir, file) for file in outpatient_files]

    DXVER_dict = {'9': icd9_to_css, '0': icd10_to_css}

    for file in files:
        inpat = pd.read_csv(file, dtype=str)

        DATE_NAME = [col for col in inpat.columns if 'DATE' in col][0]

        inpat = inpat[inpat['ENROLID'].isin(list(patient_list))]
        inpat = inpat[~inpat[DATE_NAME].isnull()]

        DX_col = [col for col in inpat.columns if 'DX' in col]

        for index, row in tqdm(inpat.iterrows(), total=len(inpat)):
            dxs = list(row[DX_col])
            enrolid = row['ENROLID']
            date = row[DATE_NAME]
            DXVER = '9'
            if dxs:
                if 'DXVER' in inpat.columns:  # files after 2015: a mix usage of both ICD-9 codes andd ICD-10 codes;
                    DXVER = row['DXVER']
                dxs = get_css_code_for_icd(dxs, DXVER_dict[DXVER])
                if enrolid not in user_dx:
                    user_dx[enrolid][date] = dxs
                else:
                    if date not in user_dx[enrolid]:
                        user_dx[enrolid][date] = dxs
                    else:
                        user_dx[enrolid][date].extend(dxs)

    return user_dx


def get_css_code_for_icd(icd_codes, icd_to_css):
    css_codes = []
    for icd_code in icd_codes:
        if not pd.isnull(icd_code):
            for i in range(len(icd_code), -1, -1):
                if icd_code[:i] in icd_to_css:
                    css_codes.append(icd_to_css.get(icd_code[:i]))
                    break

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


    return user_cohort_dx


class AutoVivification(dict):
    """Implementation of perl's autovivification feature."""
    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value


def get_user_cohort_dx(indir, cad_prescription_taken_by_patient, icd9_to_css, icd10_to_css, min_patient, patient_list):
    user_dx = get_user_dx(indir, patient_list, icd9_to_css, icd10_to_css)
    return pre_user_cohort_dx(user_dx, cad_prescription_taken_by_patient, min_patient)

