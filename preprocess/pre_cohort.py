from collections import defaultdict
from datetime import  datetime
import pickle


def get_patient_cohort(root_file):
    patient_1stDX_date = {}
    patient_start_date = {}

    for dir in ['CAD2012', 'CAD2013-2016']:
        file = root_file + dir + '/Cohort.csv'
        with open(file, 'r') as f:
            next(f)
            for row in f:
                row = row.split(',')
                enrolid, dx_date, start_date = row[0], row[3], row[4]
                patient_1stDX_date[enrolid] = datetime.strptime(dx_date, '%m/%d/%Y')
                patient_start_date[enrolid] = datetime.strptime(start_date, '%m/%d/%Y')

    my_dump(patient_1stDX_date, '../pickles/patient_1stDX_data.pkl')
    my_dump(patient_start_date, '../pickles/patient_start_date.pkl')
    return patient_1stDX_date, patient_start_date

# 1. exclude patients whose index date prior to 1st DX
# 2. exclude patients who fail to constantly take the drug within 730 days (interval <= 90)
# 3. exclude patients whose baseline period (time before index date) is less than 365 days

def exclude(cad_prescription_taken_by_patient, patient_1stDX_date, patient_start_date, interval,
            followup, baseline):

    cad_prescription_taken_by_patient_exclude = defaultdict(dict)
    cad_patient_take_prescription_exclude = defaultdict(dict)

    for drug, taken_by_patient in cad_prescription_taken_by_patient.items():
        for patient, take_times in taken_by_patient.items():
            dates = [datetime.strptime(date, '%m/%d/%Y') for (date, days) in take_times if date and days]
            dates = sorted(dates)
            dates_days = {datetime.strptime(date, '%m/%d/%Y'): int(days) for (date, days) in take_times if
                          date and days}
            DX = patient_1stDX_date.get(patient, datetime.max)
            index_date = dates[0]
            start_date = patient_start_date.get(patient, datetime.max)
            if criteria_1_is_valid(index_date, DX) and criteria_2_is_valid(dates, interval, followup,
                                                                           dates_days) and criteria_3_is_valid(
                    index_date, start_date, baseline):
                cad_prescription_taken_by_patient_exclude[drug][patient] = dates
                cad_patient_take_prescription_exclude[patient][drug] = dates


    # my_dump(cad_prescription_taken_by_patient_exclude, dump_file_prescription)
    # my_dump(cad_patient_take_prescription_exclude, dump_file_patient)

    return cad_prescription_taken_by_patient_exclude, cad_patient_take_prescription_exclude


def criteria_1_is_valid(index_date, DX):
    return (index_date - DX).days > 0


def criteria_2_is_valid(dates, interval, followup, dates_days):
    if (dates[-1] - dates[0]).days <= (followup - 89):
        return False
    for i in range(1, len(dates)):
        sup_day = dates_days.get(dates[i - 1])
        if (dates[i] - dates[i - 1]).days - sup_day > interval:
            return False

    return True


def criteria_3_is_valid(index_date, start_date, baseline):
    return (index_date - start_date).days >= baseline


def user_cohort_extractor(cad_prescription_taken_by_patient, n_patients, n_prescriptions, time_interval):

    cad_prescription_taken_by_patient_small = defaultdict(dict)
    print('number of drugs: {}'.format(len(cad_prescription_taken_by_patient)), flush=True)
    for drug, patient_take_times in cad_prescription_taken_by_patient.items():
        patient_take_times = cad_prescription_taken_by_patient.get(drug)

        if minimal_criteria_is_valid(patient_take_times, n_patients, time_interval, n_prescriptions):
            cad_prescription_taken_by_patient[drug] = patient_take_times

    return cad_prescription_taken_by_patient_small
    # my_dump(cad_prescription_taken_by_patient_small, dump_file)


def minimal_criteria_is_valid(patient_take_times, n_patients, time_interval, n_prescriptions):
    if len(patient_take_times) < n_patients:
        return False

    count = 0
    for patient, take_times in patient_take_times.items():
        if drug_time_interval_is_valid(take_times, n_prescriptions, time_interval):
            count += 1
        if count >= n_patients:
            return True

    return False


def drug_time_interval_is_valid(take_times, n_prescription, time_interval):
    count = 0
    dates = [datetime.strptime(pair[0], '%m/%d/%Y') for pair in take_times if pair[0] and pair[1]]
    dates = sorted(dates)
    for i in range(1, len(dates)):
        if (dates[i] - dates[i-1]).days >= time_interval:
            count += 2
        if count >= n_prescription:
            return True
    return False


def my_dump(obj, file_name):
    print('dumping...', flush=True)
    pickle.dump(obj, open(file_name, 'wb'))
    print('dumped...', flush=True)


def my_load(file_name):

    print('loading...', flush=True)
    return pickle.load(open(file_name, 'rb'))


if __name__ == '__main__':

    # cad_prescription_taken_by_patient = pickle.load(open('../pickles/cad_prescription_taken_by_patient.pkl', 'rb'))
    # n_patients = 500
    # n_prescriptions = 2
    # time_interval = 30
    # dump_file = 'cad_prescription_taken_by_patient_small'
    # user_cohort_extractor(cad_prescription_taken_by_patient, n_patients, n_prescriptions, time_interval,
    #                       dump_file)

    # cad_prescription_taken_by_patient = pickle.load(open('../pickles/cad_prescription_taken_by_patient.pkl', 'rb'))
    #
    root_file = '../data/'
    patient_1stDX_date, patient_start_date = get_patient_cohort(root_file)
    #
    # interval = 7
    # followup = 730
    # baseline = 365
    # dump_file_prescription = '../res/8.7/cad_prescription_taken_by_patient_exclude.pkl'
    # dump_file_patient = '../res/8.7/cad_patient_take_prescription_exclude.pkl'
    # exclude(cad_prescription_taken_by_patient, patient_1stDX_date, patient_start_date, interval,
    #         followup, baseline, dump_file_prescription, dump_file_patient)

    # cad_prescription_taken_by_patient_exclude = my_load('../res/8.7/cad_prescription_taken_by_patient_exclude_30.pkl')
    # n_drug_with_500_patients = [drug for drug, patients in cad_prescription_taken_by_patient_exclude.items() if len(patients.keys())>=500]
    # print(len(n_drug_with_500_patients))