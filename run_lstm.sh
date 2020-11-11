#!/usr/bin/env bash
work_path=$(cd "dirname $0"; PWD)
FILES="${work_path}/preprocess/save_cohort_all/*.pkl"
OUTPUT_LSTM="${work_path}/deep-ipw/log/log_LSTM"
OUTPUT_LR="${work_path}/deep-ipw/log/log_LR"
SAVE="${work_path}/deep-ipw/save/save_model"
SAVE_DB="${work_path}/deep-ipw/save/save_db"
PICKLES="${work_path}/deep-ipw/pickles"

mkdir -p ${OUTPUT_LSTM} ${OUTPUT_LR} ${SAVE} ${SAVE_DB}

for f in $FILES
do
  data_dir=${f:0:61}
  drug_id=${f:61:-4}
  outputs_lstm="${OUTPUT_LSTM}/${drug_id}.txt"
  outputs_lr="${OUTPUT_LR}/${drug_id}.txt"
  save_model="${SAVE}/${drug_id}.pt"
  save_db="${SAVE_DB}/${drug_id}.db"
  seed=99
  python ${work_path}/deep-ipw/main.py --data_dir ${data_dir} --controlled_drug random --pickles_dir ${PICKLES} --treated_drug_file ${drug_id} --save_model_filename ${save_model} --random_seed ${seed} --outputs_lstm ${outputs_lstm} --outputs_lr ${outputs_lr} --save_db ${save_db}
done
