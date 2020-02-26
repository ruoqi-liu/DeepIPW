import torch
import numpy as np
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def transfer_data(model, dataloader, cuda=True):
    with torch.no_grad():
        model.eval()
        loss_treatment=[]
        logits_treatment = []
        labels_treatment = []
        labels_outcome = []
        original_val = []
        for confounder,treatment,outcome in dataloader:
            if cuda:
                confounder[0] = confounder[0].to('cuda')
                confounder[1] = confounder[1].to('cuda')
                confounder[2] = confounder[2].to('cuda')
                confounder[3] = confounder[3].to('cuda')
                treatment = treatment.to('cuda')
                outcome = outcome.to('cuda')

            treatment_logits, original = model(confounder)
            loss_t = F.binary_cross_entropy_with_logits(treatment_logits, treatment.float())

            if cuda:
                logits_t = treatment_logits.to('cpu').detach().data.numpy()
                labels_t = treatment.to('cpu').detach().data.numpy()
                original = original.to('cpu').detach().data.numpy()
                labels_o = outcome.to('cpu').detach().data.numpy()
            else:
                logits_t = treatment_logits.detach().data.numpy()
                labels_t = treatment.detach().data.numpy()
                original = original.detach().data.numpy()
                labels_o = outcome.detach().data.numpy()

            logits_treatment.append(logits_t)
            labels_treatment.append(labels_t)
            labels_outcome.append(labels_o)
            loss_treatment.append(loss_t.item())
            original_val.extend(original)

        loss_treatment = np.mean(loss_treatment)

        golds_treatment = np.concatenate(labels_treatment)
        golds_outcome = np.concatenate(labels_outcome)
        logits_treatment = np.concatenate(logits_treatment)

        return loss_treatment, golds_treatment, logits_treatment, golds_outcome, original_val


def model_eval(model, dataloader, normalized=False, cuda=True):

    loss_treatment, golds_treatment, logits_treatment, golds_outcome, original_val = transfer_data(model, dataloader, cuda=cuda)

    AUC_treatment = cal_AUC(logits_treatment,golds_treatment)

    max_unbalanced = cal_deviation(original_val, golds_treatment, logits_treatment, normalized)

    ATE = cal_ATE(golds_treatment, logits_treatment, golds_outcome, normalized)

    return loss_treatment, AUC_treatment, max_unbalanced, ATE


def _model_eval(model, dataloader, normalized=False, cuda=True):

    _, golds_treatment, logits_treatment, _, original_val = transfer_data(model, dataloader, cuda=cuda)

    max_unbalanced = cal_deviation(original_val, golds_treatment, logits_treatment, normalized)

    return golds_treatment, logits_treatment, max_unbalanced

def cal_AUC(y_pred_prob,y_true):

    AUC=roc_auc_score(y_true,y_pred_prob)

    return AUC


def cal_weights(golds_treatment, logits_treatment, normalized):
    ones_idx, zeros_idx = np.where(golds_treatment == 1), np.where(golds_treatment == 0)

    if not normalized:
        logits_treatment = 1 / (1 + np.exp(-logits_treatment))
    p_T = len(ones_idx[0]) / (len(ones_idx[0]) + len(zeros_idx[0]))
    treated_w, controlled_w = p_T / logits_treatment[ones_idx], (1 - p_T) / (1. - logits_treatment[zeros_idx])

    treated_w = np.clip(treated_w, a_min=1e-06, a_max=100)
    controlled_w = np.clip(controlled_w, a_min=1e-06, a_max=100)

    treated_w, controlled_w = np.reshape(treated_w, (len(treated_w), 1)), np.reshape(controlled_w,
                                                                                     (len(controlled_w), 1))
    return treated_w, controlled_w


def cal_deviation(hidden_val, golds_treatment, logits_treatment, normalized):
    ones_idx, zeros_idx = np.where(golds_treatment == 1), np.where(golds_treatment == 0)

    treated_w, controlled_w = cal_weights(golds_treatment, logits_treatment, normalized=normalized)

    hidden_val = np.asarray(hidden_val)
    hidden_treated, hidden_controlled = hidden_val[ones_idx], hidden_val[zeros_idx]

    hidden_treated_w, hidden_controlled_w = np.multiply(hidden_treated, treated_w), np.multiply(hidden_controlled,
                                                                                                controlled_w)

    if not normalized:
        hidden_treated_mu, hidden_treated_var = np.mean(hidden_treated, axis=0), np.var(hidden_treated, axis=0)
        hidden_controlled_mu, hidden_controlled_var = np.mean(hidden_controlled, axis=0), np.var(hidden_controlled,
                                                                                                 axis=0)

    else:
        hidden_controlled_mu = np.mean(hidden_controlled, axis=0)
        hidden_controlled_var = hidden_controlled_mu * (1 - hidden_controlled_mu)
        hidden_treated_mu = np.mean(hidden_treated, axis=0)
        hidden_treated_var = hidden_treated_mu * (1 - hidden_treated_mu)

    VAR = np.sqrt((hidden_treated_var + hidden_controlled_var) / 2)
    hidden_deviation = np.abs(hidden_treated_mu - hidden_controlled_mu) / VAR
    hidden_deviation[np.isnan(hidden_deviation)] = 0
    max_unbalanced_original = np.max(hidden_deviation)

    hidden_treated_w_mu, hidden_treated_w_var = np.mean(hidden_treated_w, axis=0), np.var(hidden_treated_w, axis=0)
    hidden_controlled_w_mu, hidden_controlled_w_var = np.mean(hidden_controlled_w, axis=0), np.var(hidden_controlled_w,
                                                                                                   axis=0)

    # hidden_controlled_w_mu = np.mean(hidden_controlled_w, axis=0)
    # hidden_controlled_w_var = hidden_controlled_w_mu * (1 - hidden_controlled_w_mu)
    # hidden_treated_w_mu = np.mean(hidden_treated_w, axis=0)
    # hidden_treated_w_var = hidden_treated_w_mu * (1 - hidden_treated_w_mu)

    VAR = np.sqrt((hidden_treated_w_var + hidden_controlled_w_var) / 2)
    hidden_deviation_w = np.abs(hidden_treated_w_mu - hidden_controlled_w_mu) / VAR
    hidden_deviation_w[np.isnan(hidden_deviation_w)] = 0
    max_unbalanced_weighted = np.max(hidden_deviation_w)

    return max_unbalanced_original, hidden_deviation, max_unbalanced_weighted, hidden_deviation_w


def plot(hidden_treated, hidden_controlled, save_file):

    tsne = TSNE(n_components=2)

    treated_embedded = tsne.fit_transform(hidden_treated)
    controlled_embedded = tsne.fit_transform(hidden_controlled)

    plt.figure()

    treated_x, treated_y = treated_embedded[:,0], treated_embedded[:,1]
    controlled_x, controlled_y = controlled_embedded[:,0], controlled_embedded[:,1]

    plt.scatter(treated_x, treated_y, alpha=0.8, c='red', edgecolors='none', s=30, label='treated')
    plt.scatter(controlled_x, controlled_y, alpha=0.8, c='blue', edgecolors='none', s=30, label='controlled')

    plt.legend(loc=2)
    plt.savefig(save_file)



def cal_ATE(golds_treatment, logits_treatment, golds_outcome, normalized):
    ones_idx, zeros_idx = np.where(golds_treatment == 1), np.where(golds_treatment == 0)

    treated_w, controlled_w = cal_weights(golds_treatment, logits_treatment, normalized)

    treated_outcome, controlled_outcome = golds_outcome[ones_idx], golds_outcome[zeros_idx]
    treated_outcome_w, controlled_outcome_w = np.multiply(treated_outcome, treated_w.reshape(
        len(treated_w))), np.multiply(controlled_outcome, controlled_w.reshape(len(controlled_w)))

    UncorrectedEstimator_EY1_val, UncorrectedEstimator_EY0_val = np.mean(treated_outcome), np.mean(controlled_outcome)
    ATE = UncorrectedEstimator_EY1_val - UncorrectedEstimator_EY0_val
    IPWEstimator_EY1_val, IPWEstimator_EY0_val = np.mean(treated_outcome_w), np.mean(controlled_outcome_w)
    ATE_w = IPWEstimator_EY1_val - IPWEstimator_EY0_val

    return (UncorrectedEstimator_EY1_val, UncorrectedEstimator_EY0_val, ATE), (
        IPWEstimator_EY1_val, IPWEstimator_EY0_val, ATE_w)