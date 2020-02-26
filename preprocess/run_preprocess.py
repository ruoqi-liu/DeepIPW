import argparse
import os

from preprocess.pre_cohort import exclude
from preprocess.pre_cohort_rx import pre_user_cohort_rx_v2
from preprocess.pre_cohort_dx import get_user_cohort_dx
from preprocess.pre_demo import get_user_cohort_demo
from preprocess.pre_outcome import get_patient_list, pre_user_cohort_outcome
from preprocess.user_cohort import pre_user_cohort_triplet
import pickle


def parse_args():
    parser = argparse.ArgumentParser(description='process parameters')
    parser.add_argument('--min_patients', default=500, help='minimum number of patients for each cohort.')
    parser.add_argument('--min_prescription', default=2,  help='minimum times of prescriptions of each drug.')
    parser.add_argument('--time_interval', default=30, help='minimum time interval for every two prescriptions')
    parser.add_argument('--followup', default=730, help='number of days of followup period')
    parser.add_argument('--baseline', default=365, help='number of days of baseline period')
    parser.add_argument('--input_pickles', default=os.path.join(os.getcwd(), 'pickles/'))
    parser.add_argument('--save_cohort_all', required=True, default=os.path.join(os.getcwd(), 'tmp/save_cohort_all/'))

    args = parser.parse_args()
    return args


def main(args):
    cad_prescription_taken_by_patient = pickle.load(
        open(os.path.join(args.input_pickles, 'cad_prescription_taken_by_patient.pkl'), 'rb'))
    patient_1stDX_date = pickle.load(open(os.path.join(args.input_pickles, 'patient_1stDX_date.pkl'), 'rb'))
    patient_start_date = pickle.load(open(os.path.join(args.input_pickles, 'patient_start_date.pkl'), 'rb'))
    icd9_to_css = pickle.load(open(os.path.join(args.input_pickles, 'icd9_to_css.pkl'), 'rb'))
    icd10_to_css = pickle.load(open(os.path.join(args.input_pickles, 'icd10_to_css.pkl'), 'rb'))

    save_prescription, save_patient = exclude(cad_prescription_taken_by_patient, patient_1stDX_date,
                                                   patient_start_date, args.time_interval,
                                                   args.followup, args.baseline)

    save_cohort_rx = pre_user_cohort_rx_v2(save_prescription, save_patient, args.min_patients)
    save_cohort_dx = get_user_cohort_dx(save_prescription, icd9_to_css, icd10_to_css, args.min_patients)
    save_cohort_demo = get_user_cohort_demo(save_prescription, args.min_patients)

    patient_list = get_patient_list(args.min_patients, save_prescription)
    save_cohort_outcome = {}
    # Heart Failure
    codes9 = ['425', '428', '40201', '40211', '40291', '40401', '40403', '40411', '40413', '40491', '40493', 'K77']
    codes0 = ['I11', 'I13', 'I50', 'I42', 'K77']
    save_cohort_outcome['heart-failure'] = pre_user_cohort_outcome(patient_list, codes9, codes0)

    # Stroke
    codes9 = ['4380', '4381', '4382', '4383', '4384', '4385', '4386', '4387', '4388', '4389', 'V1254']
    codes0 = ['Z8673', 'I60', 'I61', 'I62', 'I63', 'I64', 'I65', 'I66', 'I67', 'I68', 'I69', 'G458', 'G459']
    save_cohort_outcome['stroke'] = pre_user_cohort_outcome(patient_list, codes9, codes0)




    pre_user_cohort_triplet(save_prescription, save_cohort_rx, save_cohort_dx,
                            save_cohort_outcome, save_cohort_demo, args.save_cohort_all)

    pre_user_cohort_triplet(cad_prescription_taken_by_patient, save_cohort_rx, save_cohort_dx,
                            save_cohort_outcome,
                            save_cohort_demo, args.save_cohort_all)



if __name__ == '__main__':
    main(args=parse_args())
