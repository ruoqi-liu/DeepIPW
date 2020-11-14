from datetime import  datetime
import pickle
from tqdm import tqdm


def pre_user_cohort_triplet(cad_prescription_taken_by_patient, cad_user_cohort_rx, cad_user_cohort_dx,
                            save_cohort_outcome,
                    cad_user_cohort_demo, out_file_root):
    cohorts_size = dict()
    for drug, taken_by_patient in tqdm(cad_user_cohort_rx.items()):
        file_x = '{}/{}.pkl'.format(out_file_root, drug)
        triples = []
        for patient, taken_times in taken_by_patient.items():
            index_date = cad_prescription_taken_by_patient.get(drug).get(patient)[0]

            dx = cad_user_cohort_dx.get(drug).get(patient)

            demo = cad_user_cohort_demo.get(patient)
            demo_feature_vector =get_demo_feature_vector(demo, index_date)

            outcome_feature_vector = []
            for outcome_name, outcome_map in save_cohort_outcome.items():

                outcome_dates = outcome_map.get(patient, [])
                dates = [datetime.strptime(date.strip('\n'), '%m/%d/%Y') for date in outcome_dates if date]
                dates = sorted(dates)
                outcome_feature_vector.append(get_outcome_feature_vector(dates, index_date))

            outcome = max(outcome_feature_vector)

            rx_codes, dx_codes = [], []
            if taken_times:
                rx_codes = [rx_code for date, rx_code in sorted(taken_times.items(), key= lambda x:x[0])]
            if dx:
                dx_codes = [list(dx_code) for date, dx_code in sorted(dx.items(), key= lambda x:x[0])]
            triple = (patient, [rx_codes, dx_codes, demo_feature_vector[0], demo_feature_vector[1]],outcome)
            triples.append(triple)

        cohorts_size['{}.pkl'.format(drug)] = len(triples)
        pickle.dump(triples, open(file_x, 'wb'))

    pickle.dump(cohorts_size, open('{}/cohorts_size.pkl'.format(out_file_root), 'wb'))


def get_outcome_feature_vector(dates, index_date):
    for date in dates:
        if date > index_date and (date - index_date).days <= 730:
            return 1
    return 0


def get_rx_feature_vector(taken_times, RX2id, size):
    feature_vector = [0] * size
    for rx in taken_times:
        if rx in RX2id:
            id = RX2id.get(rx)
            feature_vector[id] = 1

    return feature_vector


def get_dx_feature_vector(dx, CCS2id, size):

    feature_vector = [0] * size
    not_find = set()
    for code in dx:
        for c in code:
            if c in CCS2id:
                id = CCS2id.get(c)
                feature_vector[id] = 1

    return feature_vector, not_find


def get_demo_feature_vector(demo, index_date):
    if not demo:
        return [0, 0]
    db, sex = demo
    index_date_y = index_date.year
    age = index_date_y - int(db)
    sex_n = int(sex) - 1
    return [age, sex_n]


