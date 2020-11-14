from collections import defaultdict
from datetime import  datetime
import pickle
from tqdm import tqdm
import os
import pandas as pd


def is_valid_outcome_range(dx, code_range):
    for code in code_range:
        if dx.startswith(code):
            return True
    return False


def pre_user_cohort_outcome(indir, patient_list, codes9, codes0):
    cad_user_cohort_outcome = defaultdict(list)
    # data format: {patient_id: [date_of_outcome, date_of_outcome,...,date_of_outcome]}

    inpatient_dir = os.path.join(indir, 'inpatient')
    inpatient_files = os.listdir(inpatient_dir)
    outpatient_dir = os.path.join(indir, 'outpatient')
    outpatient_files = os.listdir(outpatient_dir)

    files = [os.path.join(inpatient_dir, file) for file in inpatient_files]\
            + [os.path.join(outpatient_dir, file) for file in outpatient_files]

    DXVER_dict = {'9': codes9, '0': codes0}
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
                for dx in dxs:
                    if not pd.isnull(dx):
                        if is_valid_outcome_range(dx, DXVER_dict[DXVER]):
                            cad_user_cohort_outcome[enrolid].append(date)

    return cad_user_cohort_outcome
