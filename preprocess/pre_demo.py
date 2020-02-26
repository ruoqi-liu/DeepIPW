import pickle

def pre_user_cohort_demo(patient_list):
    cad_user_cohort_demo = {}
    for dir in ['../data/CAD2012/', '../data/CAD2013-2016/']:
        print('dir: {}'.format(dir), flush=True)
        file = dir + 'demo.csv'
        with open(file, 'r') as f:
            next(f)
            for row in f:
                row = row.split(',')
                id, db, sex = row[0], row[1], row[2]
                if id in patient_list:
                    cad_user_cohort_demo[id] = (db, sex)


    # my_dump(cad_user_cohort_demo, dump_file)

    return cad_user_cohort_demo



def my_dump(obj, file_name):
    print('dumping...')
    pickle.dump(obj, open(file_name, 'wb'))

def my_load(file_name):

    print('loading...', flush=True)
    return pickle.load(open(file_name, 'rb'))


def get_patient_list(min_patient, cad_prescription_taken_by_patient):
    patients_list = set()
    for drug, patients in cad_prescription_taken_by_patient.items():
        if len(patients) >= min_patient:
            for patient in patients:
                patients_list.add(patient)
    return patients_list


def get_user_cohort_demo(cad_prescription_taken_by_patient, min_patient):
    patient_list = get_patient_list(min_patient, cad_prescription_taken_by_patient)
    return pre_user_cohort_demo(patient_list)


if __name__ == '__main__':
    cad_prescription_taken_by_patient = my_load('../res/8.7/cad_prescription_taken_by_patient_exclude_30.pkl')
    min_patient = 500
    dump_file = '../res/8.7/user_cohort_demo_30.pkl'

    get_user_cohort_demo(cad_prescription_taken_by_patient, min_patient, dump_file)

    # demo = my_load('../pickles/cad_user_cohort_demo.pkl')
    # print()

