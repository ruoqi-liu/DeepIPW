from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn import svm
from sklearn import tree

from sklearn.ensemble import AdaBoostClassifier
import numpy as np


class PropensityEstimator:
    def __init__(self, learner, confounder, treatment):
        if learner == 'Logistic-regression':
            self.learner = LogisticRegression(solver='liblinear', penalty='l2', C=1).fit(confounder, treatment)
        elif learner == 'SVM':
            self.learner = svm.SVC().fit(confounder, treatment)
        elif learner == 'CART':
            self.learner = tree.DecisionTreeClassifier(max_depth=6).fit(confounder, treatment)

    def compute_weights(self, confounder):
        pred_propensity = self.learner.predict_proba(confounder)[:,1]
        # pred_clip_propensity = np.clip(pred_propensity, a_min=np.quantile(pred_propensity, 0.1), a_max=np.quantile(pred_propensity, 0.9))
        # inverse_propensity = 1. / pred_propensity
        return pred_propensity


class OutcomeEstimator:
    def __init__(self, learner, x_input, outcome, sample_weights=None):
        if learner == 'Logistic-regression':
            self.learner = LogisticRegression(solver='liblinear', penalty='l2', C=1).fit(x_input, outcome, sample_weight=sample_weights)
        elif learner == 'SGD':
            self.learner = SGDClassifier(loss='log').fit(x_input, outcome)
        elif learner == 'AdaBoost':
            self.learner = AdaBoostClassifier().fit(x_input, outcome)

    def predict_outcome(self, x_input):
        pred_outcome = self.learner.predict_proba(x_input)[:,1]
        return pred_outcome