# DeepIPW

## 1. Introduction
A computational framework for drug repurposing from real-world data. DeepIPW: Deep Inverse Propensity Weighting.

## 2. System requirement
OS: Ubuntu 16.04

GPU: NVIDIA 1080ti (11GB memory) is **minimum** requirement. We recommend NVIDIA TITAN RTX 6000 GPUs. 

## 3. Dependencies
```
Python 3.6
Pytorch 1.2.0
Scipy 1.3.1
Numpy 1.17.2
Scikit-learn 0.22.2
```

## 4. Preprocessing data
### Dataset
The real world patient data used in this paper is [MarketScan claims data](https://www.ibm.com/products/marketscan-research-databases). Interested parties may contact IBM for acquiring the data access at https://www.ibm.com/products/marketscan-research-databases.

### Input data demo
The demo of the input data can be found in the *data* folder, where the data structures of the inputs are provided. Before running the preprocessing codes, make sure the input data format is same to the provided input demo.

For example, the data structure for the prescription table (under *data/CAD/drug/drug12.csv*) is as follows,
| ENROLID | NDCNUM | SVCDATE | DAYSUPP |
|---------|--------|---------|---------|

- ENROLID: Patient id
- NDCNUM: Drug id
- SVCDATE: Date to take the prescription
- DAYSUPP: Days supply: The number of days of drug therapy covered by this prescription

### Preprocess drug tables
```
cd preprocess
python pre_drug.py 
```

### Preprocess patient cohort
```
cd preprocess
python run_preprocess.py
```

### Parameters
- --min_patients, minimum number of patients for each cohort.
- --min_prescription, minimum times of prescriptions of each drug.
- --time_interval, minimum time interval for every two prescriptions.
- --followup, number of days of followup period.
- --baseline, number of days of baseline period.
- --input_pickles, data pickles.
- --save_cohort_all, save path.


## 5. DeepIPW model
### Bash command
```
bash run_lstm.sh
```
### Python command
```
cd deep-ipw
python main.py
```

### Parameters
- --data_dir, input cohort data
- --pickles_dir, pickles file.
- --treated_drug_file, current evaluating drug.
- --controlled_drug, sampled controlled drugs (randomly sampling or ATC class).
- --controlled_drug_ratio, ratio of the number of controlled drug.
- --input_pickles, data pickles.
- --random_seed.
- --batch_size.
- --diag_emb_size.
- --med_emb_size.
- --med_hidden_size.
- --diag_hidden_size.
- --learning_rate.
- --weight_decay.
- --epochs
- --save_model_filename.
- --outputs_lstm.
- --outputs_lr.
- --save_db.
