import numpy as np
import os
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


def pre_user_cohort(cad_prescription_taken_by_patient, cad_user_cohort_rx, cad_user_cohort_dx, cad_user_cohort_outcome,
                    cad_user_cohort_demo, RX2id, CCS2id, out_file_root):
    dx_dict_size = len(set(CCS2id.values()))
    rx_dict_size = len(set(RX2id.values()))
    for drug, taken_by_patient in tqdm(cad_user_cohort_rx.items()):

        file_x = out_file_root + '/x/' + drug + '.csv'
        os.makedirs(os.path.dirname(file_x), exist_ok=True)
        out_x = open(file_x, 'w')
        # file_outcome = out_file_root + '/outcome/' + drug + '.csv'
        # os.makedirs(os.path.dirname(file_outcome), exist_ok=True)
        # out_outcome = open(file_outcome, 'w')

        for patient, taken_times in taken_by_patient.items():
            index_date = cad_prescription_taken_by_patient.get(drug).get(patient)[0]
            rx_feature_vector = get_rx_feature_vector(taken_times, RX2id, rx_dict_size)

            dx = cad_user_cohort_dx.get(drug).get(patient)
            dx_feature_vector, not_find = get_dx_feature_vector(dx, CCS2id, dx_dict_size)

            demo = cad_user_cohort_demo.get(patient)
            demo_feature_vector =get_demo_feature_vector(demo, index_date)

            outcome_dates = cad_user_cohort_outcome.get(patient, [])
            dates = [datetime.strptime(date.strip('\n'), '%m/%d/%Y') for date in outcome_dates if date]
            dates = sorted(dates)
            outcome_feature_vector = get_outcome_feature_vector(dates, index_date)


            features = rx_feature_vector + dx_feature_vector + demo_feature_vector + outcome_feature_vector

            out_x.write(patient + ',' + ','.join(str(x) for x in features) + '\n')
            # out_outcome.write(patient + ','  + ','.join(str(x) for x in outcome_feature_vector)+ '\n')


        out_x.close()
        # out_outcome.close()


def pre_user_cohort_triplet(cad_prescription_taken_by_patient, cad_user_cohort_rx, cad_user_cohort_dx,
                            save_cohort_outcome,
                    cad_user_cohort_demo, out_file_root):
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

            rx_codes = [rx_code for date, rx_code in sorted(taken_times.items(), key= lambda x:x[0])]
            dx_codes = [list(dx_code) for date, dx_code in sorted(dx.items(), key= lambda x:x[0])]
            triple = (patient, [rx_codes, dx_codes, demo_feature_vector[0], demo_feature_vector[1]],outcome)
            triples.append(triple)

        my_dump(triples, file_x)






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



def get_ccs2id(cad_user_cohort_dx, dump_file):

    dxs_list = set()
    for patient_dxs in cad_user_cohort_dx.values():
        for dxs in patient_dxs.values():
            for dx in dxs:
                for d in dx:
                    dxs_list.add(d)

    print('n ccs group: {}'.format(len(dxs_list)))

    ccs2id = {ccs: id for id, ccs in enumerate(list(dxs_list))}

    # my_dump(ccs2id, dump_file)

    return ccs2id




def get_rx2id(cad_user_cohort_rx, dump_file):
    rx2NDC = {}
    with open('../data/NDC_complete_mapping.csv', 'r') as f:
        next(f)
        for row in f:
            row = row.split(',')
            rx, ndc = row[1], row[2]
            rx2NDC[rx] = ndc

    # 2. NDC_to_THERCLS

    NDC2THERCLS = {}
    with open('../data/redbook.csv', 'r') as f:
        next(f)
        for row in f:
            row = row.split(',')
            ndc, cls = row[0], row[12]
            NDC2THERCLS[ndc] = cls

    group_list = set()
    drug_list = set()
    for drug, taken_times in cad_user_cohort_rx.items():
        for patient, rxs in taken_times.items():
            for rx in rxs:
                ndc = rx2NDC.get(rx)
                grp = NDC2THERCLS.get(ndc)
                group_list.add(grp)
                drug_list.add(rx)

    group_list = list(group_list)
    grp2id = {cls: i for i, cls in enumerate(group_list)}

    RX2id = {}
    for rx in drug_list:
        ndc = rx2NDC.get(rx)
        grp = NDC2THERCLS.get(ndc)
        id = grp2id.get(grp)
        RX2id[rx] = id

    print('n thecls rx: {}'.format(len(set(RX2id.values()))))

    # my_dump(RX2id, dump_file)
    return RX2id

class AutoVivification(dict):
    """Implementation of perl's autovivification feature."""
    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value



# if __name__ == '__main__':
#     cad_user_cohort_dx = my_load('../res/10.15/user_cohort_dx_30.pkl')
#     # dump_file = '../res/8.7/ccs2id.pkl'
#     # CCS2id = get_ccs2id(cad_user_cohort_dx, dump_file)
#     #
#     #
#     cad_user_cohort_rx = my_load('../res/10.15/user_cohort_rx_30.pkl')
#     # dump_file = '../res/8.7/rx2id.pkl'
#     # RX2id = get_rx2id(cad_user_cohort_rx, dump_file)
#
#     cad_prescription_taken_by_patient = my_load('../res/8.7/cad_prescription_taken_by_patient_exclude_30.pkl')
#     save_cohort_outcome = {}
#     save_cohort_outcome['heart-failure'] = my_load('../res/01.29/user_cohort_outcome_30_hf.pkl')
#     save_cohort_outcome['stroke'] = my_load('../res/01.28/user_cohort_outcome_30_stroke.pkl')
#     cad_user_cohort_demo = my_load('../res/8.7/user_cohort_demo_30.pkl')
#     # RX2id = my_load('../res/8.7/rx2id.pkl')
#     # CCS2id = my_load('../res/8.7/ccs2id.pkl')
#     out_file_root = '../user_cohort/01.31/'
#     pre_user_cohort_triplet(cad_prescription_taken_by_patient, cad_user_cohort_rx, cad_user_cohort_dx,
#                             save_cohort_outcome,
#                             cad_user_cohort_demo, out_file_root)
