from datetime import datetime
import pickle
import pandas as pd


def get_patient_init_date(indir, outdir):

    patient_1stDX_date = {}
    patient_start_date = {}
    file = '{}/Cohort.csv'.format(indir)
    with open(file, 'r') as f:
        next(f)
        for row in f:
            row = row.split(',')
            enrolid, dx_date, start_date = row[0], row[1], row[2]
            patient_1stDX_date[enrolid] = datetime.strptime(dx_date, '%m/%d/%Y')
            patient_start_date[enrolid] = datetime.strptime(start_date, '%m/%d/%Y')

    out1 = '{}/patient_1stDX_data.pkl'.format(outdir)
    pickle.dump(patient_1stDX_date, open(out1, 'wb'))

    out2 = '{}/patient_start_date.pkl'.format(outdir)
    pickle.dump(patient_start_date, open(out2, 'wb'))
    return patient_1stDX_date, patient_start_date