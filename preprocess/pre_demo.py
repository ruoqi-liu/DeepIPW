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

    return cad_user_cohort_demo


def get_user_cohort_demo(indir,patient_list):
    return pre_user_cohort_demo(indir, patient_list)

