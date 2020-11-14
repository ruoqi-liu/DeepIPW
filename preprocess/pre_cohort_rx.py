from collections import defaultdict
from datetime import  datetime
import pickle
from tqdm import tqdm


def pre_user_cohort_rx(cad_prescription_taken_by_patient, cad_patient_take_prescription, min_patients):
    cad_user_cohort_rx = defaultdict(dict)

    for drug, taken_by_patient in tqdm(cad_prescription_taken_by_patient.items()):
        if len(taken_by_patient.keys()) >= min_patients:
            for patient, take_dates in taken_by_patient.items():
                index_date = take_dates[0]
                patient_prescription_list = cad_patient_take_prescription.get(patient)
                for prescription, dates_days in patient_prescription_list.items():
                    dates = [datetime.strptime(date, '%m/%d/%Y') for date, days in dates_days]
                    dates = sorted(dates)
                    if drug_is_taken_in_baseline(index_date, dates):
                        if drug not in cad_user_cohort_rx:
                            cad_user_cohort_rx[drug][patient] = [prescription]
                        else:
                            if patient not in cad_user_cohort_rx[drug]:
                                cad_user_cohort_rx[drug][patient] = [prescription]
                            else:
                                cad_user_cohort_rx[drug][patient].append(prescription)

    return cad_user_cohort_rx


def get_prescription_taken_times(index_date, dates, dates_2_days):
    cnt = 0
    for date in dates:
        if (index_date - date).days - dates_2_days[date] > 0:
            cnt += 1
        else:
            return cnt
    return cnt


# v1
def drug_is_taken_in_baseline(index_date, dates):
    for date in dates:
        if (index_date - date).days > 0:
            return True
    return False


# v2
def pre_user_cohort_rx_v2(cad_prescription_taken_by_patient, cad_patient_take_prescription, min_patients):
    cad_user_cohort_rx = AutoVivification()

    for drug, taken_by_patient in tqdm(cad_prescription_taken_by_patient.items()):
        if len(taken_by_patient.keys()) >= min_patients:
            for patient, take_dates in taken_by_patient.items():
                index_date = take_dates[0]
                patient_prescription_list = cad_patient_take_prescription.get(patient)
                for prescription, dates_days in patient_prescription_list.items():
                    # dates = [datetime.strptime(date, '%m/%d/%Y') for date, days in dates_days]
                    dates = sorted(dates_days)
                    dates = drug_is_taken_in_baseline_v2(index_date, dates)
                    if dates:
                        for date in dates:
                            if drug not in cad_user_cohort_rx:
                                cad_user_cohort_rx[drug][patient][date] = [prescription]
                            else:
                                if patient not in cad_user_cohort_rx[drug]:
                                    cad_user_cohort_rx[drug][patient][date] = [prescription]
                                else:
                                    if date not in cad_user_cohort_rx[drug][patient]:
                                        cad_user_cohort_rx[drug][patient][date] = [prescription]
                                    else:
                                        cad_user_cohort_rx[drug][patient][date].append(prescription)


    return cad_user_cohort_rx


# v2 for LSTM - save timestamp
def drug_is_taken_in_baseline_v2(index_date, dates):
    res = []
    for date in dates:
        if (index_date - date).days > 0:
            res.append(date)
    if len(res)>0:
        return res
    return False


class AutoVivification(dict):
    """Implementation of perl's autovivification feature."""
    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value

