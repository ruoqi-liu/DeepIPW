import logging
import numpy as np
import torch.utils.data
from vocab import *
from tqdm import tqdm

logger=logging.getLogger()

class Dataset(torch.utils.data.Dataset):
    def __init__(self, treated_patient_list, control_patient_list, diag_code_vocab=None, med_code_vocab=None):
        self.treated_patient_list = treated_patient_list
        self.control_patient_list = control_patient_list

        self.diagnoses_visits = []
        self.medication_visits=[]
        self.sexes=[]
        self.ages=[]

        self.outcome=[]
        self.treatment=[]


        for _, patient_confounder, patient_outcome in tqdm(self.treated_patient_list):
            self.outcome.append(patient_outcome)
            self.treatment.append(1)
            med_visit, diag_visit, age, sex = patient_confounder
            self.medication_visits.append(med_visit)
            self.diagnoses_visits.append(diag_visit)
            self.sexes.append(sex)
            self.ages.append(age)


        for _, patient_confounder, patient_outcome in tqdm(self.control_patient_list):
            self.outcome.append(patient_outcome)
            self.treatment.append(0)
            med_visit, diag_visit, age, sex = patient_confounder
            self.medication_visits.append(med_visit)
            self.diagnoses_visits.append(diag_visit)
            self.sexes.append(sex)
            self.ages.append(age)


        if diag_code_vocab is None:
            self.diag_code_vocab=CodeVocab()
            self.diag_code_vocab.add_patients_visits(self.diagnoses_visits)

        if med_code_vocab is None:
            self.med_code_vocab=CodeVocab()
            self.med_code_vocab.add_patients_visits(self.medication_visits)


        logger.info('Created Diagnoses Vocab: %s' % self.diag_code_vocab)
        logger.info('Created Medication Vocab: %s' % self.med_code_vocab)

        self.diag_visit_max_length=max([len(patient_visit) for patient_visit in self.diagnoses_visits])
        self.med_visit_max_length = max([len(patient_visit) for patient_visit in self.medication_visits])

        self.diag_vocab_length=len(self.diag_code_vocab)
        self.med_vocab_length=len(self.med_code_vocab)


        logger.info('Diagnoses Visit Max Length: %d' % self.diag_visit_max_length)
        logger.info('Medication Visit Max Length: %d' % self.med_visit_max_length)


        # self.ages=np.abs(self.ages-np.mean(self.ages))/np.var(self.ages)
        self.ages = (self.ages - np.min(self.ages)) / np.ptp(self.ages)


    def _process_visits(self, visits, max_len_visit, vocab):
        res=np.zeros((max_len_visit,len(vocab)))
        for i, visit in enumerate(visits):
            res[i]=self._process_code(vocab,visit)
        return res

    def _process_code(self, vocab, codes):
        multi_hot = np.zeros((len(vocab, )), dtype='float')
        for code in codes:
            multi_hot[vocab.code2id[code]] = 1
        return multi_hot


    def __getitem__(self, index):
        diag=self.diagnoses_visits[index]
        diag=self._process_visits(diag,self.diag_visit_max_length,self.diag_code_vocab)

        med=self.medication_visits[index]
        med=self._process_visits(med,self.med_visit_max_length,self.med_code_vocab)


        sex= self.sexes[index]
        age=self.ages[index]
        outcome=self.outcome[index]
        treatment=self.treatment[index]

        confounder=(diag,med,sex,age)

        return confounder,treatment,outcome


    def __len__(self):
        return len(self.diagnoses_visits)




