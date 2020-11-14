#!/usr/bin/env bash
FILES="preprocess/save_cohort_all/1000560.pkl"
DATA_DIR="preprocess/save_cohort_all/"
OUTPUT_LSTM="deep-ipw/log/log_LSTM_test"
OUTPUT_LR="deep-ipw/log/log_LR_test"
SAVE="deep-ipw/save/save_model_test"
SAVE_DB="deep-ipw/save/save_db_test"
PICKLES="deep-ipw/pickles"

mkdir -p ${OUTPUT_LSTM} ${OUTPUT_LR} ${SAVE} ${SAVE_DB}

for f in $FILES
do
  drug_id=${f:${#DATA_DIR}:${#f}-${#DATA_DIR}-4}
  outputs_lstm="${OUTPUT_LSTM}/${drug_id}.txt"
  outputs_lr="${OUTPUT_LR}/${drug_id}.txt"
  save_model="${SAVE}/${drug_id}.pt"
  save_db="${SAVE_DB}/${drug_id}.db"
  seed=99
  echo $drug_id
  python deep-ipw/main.py --data_dir ${DATA_DIR} --controlled_drug random --pickles_dir ${PICKLES} --treated_drug_file ${drug_id} --save_model_filename ${save_model} --random_seed ${seed} --outputs_lstm ${outputs_lstm} --outputs_lr ${outputs_lr} --save_db ${save_db}
done
