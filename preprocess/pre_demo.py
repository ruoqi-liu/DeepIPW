import pickle

def pre_user_cohort_demo(indir, patient_list):
    cad_user_cohort_demo = {}
    file = '{}/demo.csv'.format(indir)
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



def get_user_cohort_demo(indir, cad_prescription_taken_by_patient, min_patient,patient_list):
    return pre_user_cohort_demo(indir, patient_list)


# if __name__ == '__main__':
#     cad_prescription_taken_by_patient = my_load('../res/8.7/cad_prescription_taken_by_patient_exclude_30.pkl')
#     min_patient = 500
#     dump_file = '../res/8.7/user_cohort_demo_30.pkl'


    # demo = my_load('../pickles/cad_user_cohort_demo.pkl')
    # print()

