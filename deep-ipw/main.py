from dataset import *
import pickle
import argparse
from model import *
from torch.utils.data.sampler import SubsetRandomSampler
from evaluation import model_eval, cal_deviation, cal_ATE
import torch.nn.functional as F
from baselines import *
import os
from utils import save_model, load_model
from sklearn.metrics import accuracy_score,roc_auc_score



def parse_args():
    parser = argparse.ArgumentParser(description='process parameters')
    parser.add_argument('--data_dir', type=str, default='../user_cohort/')
    parser.add_argument('--pickles_dir', type=str, default='pickles')
    parser.add_argument('--treated_drug_file', type=str)
    parser.add_argument('--controlled_drug', choices=['ATC','random'], default='random')
    parser.add_argument('--controlled_drug_ratio', type=int, default=3)
    parser.add_argument("--random_seed", type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--diag_emb_size', type=int, default=128)
    parser.add_argument('--med_emb_size', type=int, default=128)
    parser.add_argument('--med_hidden_size', type=int, default=64)
    parser.add_argument('--diag_hidden_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.000001)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--save_model_filename', type=str, default='tmp/1346823.pt')
    parser.add_argument('--outputs_lstm', type=str)
    parser.add_argument('--outputs_lr', type=str)
    parser.add_argument('--save_db', type=str)
    args = parser.parse_args()

    return args



def main(args):
    args.cuda = torch.cuda.is_available()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('args: ', args)

    np.random.seed(args.random_seed)
    output_lstm = open(args.outputs_lstm, 'a')

    treated = pickle.load(open(args.data_dir + args.treated_drug_file + '.pkl', 'rb'))
    controlled = []

    if args.controlled_drug == 'random':
        cohort_size = pickle.load(open(os.path.join(args.data_dir, 'cohorts_size.pkl'), 'rb'))
        controlled_drugs = list(set(os.listdir(args.data_dir)) - set(args.treated_drug_file+'.pkl'))
        np.random.shuffle(controlled_drugs)
        n_control_patient = 0
        controlled_drugs_range = []
        n_treat_patient = cohort_size.get(args.treated_drug_file+'.pkl')
        for c_id in controlled_drugs:
            n_control_patient += cohort_size.get(c_id)
            controlled_drugs_range.append(c_id)
            if n_control_patient >= (args.controlled_drug_ratio+1) * n_treat_patient:
                break


    else:
        ATC2DRUG = pickle.load(open(os.path.join(args.pickles_dir,'ATC2DRUG.pkl'), 'rb'))
        DRUG2ATC = pickle.load(open(os.path.join(args.pickles_dir,'DRUG2ATC.pkl'), 'rb'))

        drug_atc = DRUG2ATC.get(args.treated_drug_file)
        atc_group = set()
        for atc in drug_atc:
            for drug in ATC2DRUG.get(atc):
                atc_group.add(drug)
        if len(atc_group) > 1:
            controlled_drugs_range = [drug+'.pkl' for drug in atc_group if drug !=args.treated_drug_file]
        else:
            all_atc = set(ATC2DRUG.keys())-set(drug_atc)
            sample_atc = [atc for atc in list(all_atc) if len(ATC2DRUG.get(atc))==1]
            sample_drug  = set()
            for atc in sample_atc:
                for drug in ATC2DRUG.get(atc):
                    sample_drug.add(drug)
            controlled_drugs_range = [drug + '.pkl' for drug in sample_drug if drug != args.treated_drug_file]

    for c_drug_id in controlled_drugs_range:
        c = pickle.load(open(args.data_dir + c_drug_id, 'rb'))
        controlled.extend(c)

    intersect = set(np.asarray(treated)[:, 0]).intersection(set(np.asarray(controlled)[:, 0]))
    controlled = np.asarray([controlled[i] for i in range(len(controlled)) if controlled[i][0] not in intersect])

    controlled_indices = list(range(len(controlled)))
    controlled_sample_index = int(args.controlled_drug_ratio * len(treated))

    np.random.shuffle(controlled_indices)

    controlled_sample_indices = controlled_indices[:controlled_sample_index]

    controlled_sample = controlled[controlled_sample_indices]

    n_user, n_nonuser = len(treated), len(controlled_sample)
    print('user: {}, non_user: {}'.format(n_user, n_nonuser), flush=True)

    print("Constructed Dataset.", flush=True)
    my_dataset = Dataset(treated, controlled_sample)

    train_ratio = 0.7
    val_ratio = 0.1

    dataset_size = len(my_dataset)
    indices = list(range(dataset_size))
    train_index = int(np.floor(train_ratio * dataset_size))
    val_index = int(np.floor(val_ratio * dataset_size))

    np.random.shuffle(indices)

    train_indices, val_indices, test_indices = indices[:train_index], \
                                               indices[train_index:train_index + val_index], \
                                               indices[train_index + val_index:]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = torch.utils.data.DataLoader(my_dataset, batch_size=args.batch_size,
                                               sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(my_dataset, batch_size=args.batch_size,
                                             sampler=val_sampler)
    test_loader = torch.utils.data.DataLoader(my_dataset, batch_size=args.batch_size,
                                              sampler=test_sampler)

    data_loader = torch.utils.data.DataLoader(my_dataset, batch_size=args.batch_size,
                                              sampler=SubsetRandomSampler(indices))

    model_params = dict(
        med_hidden_size=args.med_hidden_size,
        diag_hidden_size=args.diag_hidden_size,
        hidden_size=100,
        bidirectional=True,
        med_vocab_size=len(my_dataset.med_code_vocab),
        diag_vocab_size=len(my_dataset.diag_code_vocab),
        diag_embedding_size=args.diag_emb_size,
        med_embedding_size=args.med_emb_size,
        end_index=my_dataset.diag_code_vocab[CodeVocab.END_CODE],
        pad_index=my_dataset.diag_code_vocab[CodeVocab.PAD_CODE],
    )
    print(model_params, flush=True)

    model = LSTMModel(**model_params)

    if args.cuda:
        model = model.to('cuda')

    print(model, flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # highest_auc = 0
    lowest_std = float('inf')
    # lowest_n_unbalanced = float('inf')
    for epoch in range(args.epochs):
        epoch_losses_ipw = []
        for confounder,treatment,outcome in tqdm(train_loader):
            model.train()

            # train IPW
            optimizer.zero_grad()

            if args.cuda:
                confounder[0] =confounder[0].to('cuda')
                confounder[1] = confounder[1].to('cuda')
                confounder[2] = confounder[2].to('cuda')
                confounder[3] = confounder[3].to('cuda')
                treatment =treatment.to('cuda')

            treatment_logits, _ = model(confounder)
            loss_ipw = F.binary_cross_entropy_with_logits(treatment_logits, treatment.float())

            loss_ipw.backward()
            optimizer.step()
            epoch_losses_ipw.append(loss_ipw.item())

        epoch_losses_ipw = np.mean(epoch_losses_ipw)

        print('Epoch: {}, IPW train loss: {}'.format(epoch, epoch_losses_ipw), flush=True)

        loss_val, AUC_val, max_unbalanced, ATE = model_eval(model,val_loader, cuda=args.cuda)
        _, hidden_deviation, max_unbalanced_weighted, hidden_deviation_w = max_unbalanced
        print('Val loss_treament: {}'.format(loss_val), flush=True)
        print('Val AUC_treatment: {}'.format(AUC_val), flush=True)
        print('Val Max_unbalanced: {}'.format(max_unbalanced_weighted), flush=True)
        print('ATE_w: {}'.format(ATE[1][2]), flush=True)
        if max_unbalanced_weighted < lowest_std:
            save_model(model, args.save_model_filename, model_params=model_params)
            lowest_std = max_unbalanced_weighted

        if epoch % 5 ==0:
            loss_test, AUC_test, _, _ =model_eval(model, test_loader, cuda=args.cuda)
            print('Test loss_treament: {}'.format(loss_test))
            print('Test AUC_treatment: {}'.format(AUC_test))


    mymodel = load_model(LSTMModel, args.save_model_filename)
    mymodel.to(args.device)
    _, AUC, max_unbalanced, ATE = model_eval(mymodel, data_loader, cuda=args.cuda)
    max_unbalanced_original, hidden_deviation, max_unbalanced_weighted, hidden_deviation_w = max_unbalanced

    n_unbalanced_feature = len(np.where(hidden_deviation > 0.1)[0])
    n_unbalanced_feature_w = len(np.where(hidden_deviation_w > 0.1)[0])
    n_feature = my_dataset.med_vocab_length + my_dataset.diag_vocab_length + 2

    UncorrectedEstimator_EY1_val, UncorrectedEstimator_EY0_val, ATE_original = ATE[0]
    IPWEstimator_EY1_val, IPWEstimator_EY0_val, ATE_weighted = ATE[1]
    print('AUC_treatment: {}'.format(AUC), flush=True)
    print('max_unbalanced_ori: {}, max_unbalanced_wei: {}'.format(max_unbalanced_original,
                                                                            max_unbalanced_weighted), flush=True)
    print('ATE_ori: {}, ATE_wei: {}'.format(ATE_original, ATE_weighted), flush=True)

    print('n_unbalanced_feature: {}, n_unbalanced_feature_w: {}'.format(n_unbalanced_feature, n_unbalanced_feature_w), flush=True)

    output_lstm.write('{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(n_user, n_nonuser,
                                                                max_unbalanced_original,max_unbalanced_weighted,
                                                                n_unbalanced_feature,n_unbalanced_feature_w,n_feature,
                                                                UncorrectedEstimator_EY1_val,
                                                                UncorrectedEstimator_EY0_val,
                                                                ATE_original, IPWEstimator_EY1_val,
                                                                IPWEstimator_EY0_val,
                                                                ATE_weighted))

    train_x, train_t, train_y = [], [], []
    for idx in train_indices:
        confounder, treatment, outcome = my_dataset[idx][0], my_dataset[idx][1], my_dataset[idx][2]
        dx, rx, sex, age = confounder[0], confounder[1], confounder[2], confounder[3]
        dx, rx = np.sum(dx, axis=0), np.sum(rx, axis=0)
        dx = np.where(dx > 0, 1, 0)
        rx = np.where(rx > 0, 1, 0)

        train_x.append(np.concatenate((dx, rx, [sex], [age])))
        train_t.append(treatment)
        train_y.append(outcome)

    train_x, train_t, train_y = np.asarray(train_x), np.asarray(train_t), np.asarray(train_y)

    val_x, val_t, val_y = [], [], []
    for idx in val_indices:
        confounder, treatment, outcome = my_dataset[idx][0], my_dataset[idx][1], my_dataset[idx][2]
        dx, rx, sex, age = confounder[0], confounder[1], confounder[2], confounder[3]
        dx, rx = np.sum(dx, axis=0), np.sum(rx, axis=0)
        dx = np.where(dx > 0, 1, 0)
        rx = np.where(rx > 0, 1, 0)

        val_x.append(np.concatenate((dx, rx, [sex], [age])))
        val_t.append(treatment)
        val_y.append(outcome)

    val_x, val_t, val_y = np.asarray(val_x), np.asarray(val_t), np.asarray(val_y)

    test_x, test_t, test_y = [], [], []
    for idx in test_indices:
        confounder, treatment, outcome = my_dataset[idx][0], my_dataset[idx][1], my_dataset[idx][2]
        dx, rx, sex, age = confounder[0], confounder[1], confounder[2], confounder[3]
        dx, rx = np.sum(dx, axis=0), np.sum(rx, axis=0)
        dx = np.where(dx > 0, 1, 0)
        rx = np.where(rx > 0, 1, 0)

        test_x.append(np.concatenate((dx, rx, [sex], [age])))
        test_t.append(treatment)
        test_y.append(outcome)

    test_x, test_t, test_y = np.asarray(test_x), np.asarray(test_t), np.asarray(test_y)

    output_lr = open(args.outputs_lr, 'a')

    # Compute IPW using PropensityEstimator
    pe = PropensityEstimator(learner='Logistic-regression', confounder=train_x, treatment=train_t)
    val_propensity = pe.compute_weights(confounder=val_x)
    print('LR: Val AUC_treatment: {}'.format(roc_auc_score(val_t, val_propensity)))

    x, t, y = [], [], []
    for idx in indices:
        confounder, treatment, outcome = my_dataset[idx][0], my_dataset[idx][1], my_dataset[idx][2]
        dx, rx, sex, age = confounder[0], confounder[1], confounder[2], confounder[3]
        dx, rx = np.sum(dx, axis=0), np.sum(rx, axis=0)
        dx = np.where(dx > 0, 1, 0)
        rx = np.where(rx > 0, 1, 0)

        x.append(np.concatenate((dx, rx, [sex], [age])))
        t.append(treatment)
        y.append(outcome)

    x, t, y = np.asarray(x), np.asarray(t), np.asarray(y)

    # Compute ATE using Inverse Propensity
    propensity = pe.compute_weights(confounder=x)
    AUC = roc_auc_score(t, propensity)
    print('LR: AUC_treatment: {}'.format(AUC))


    max_unbalanced_original, hidden_deviation, max_unbalanced_weighted, hidden_deviation_w = cal_deviation(
        hidden_val=x, golds_treatment=t,
        logits_treatment=propensity, normalized=True)
    print('LR: max_unbalanced_original: {}, max_unbalanced_weighted: {}'.format(max_unbalanced_original,
                                                                                      max_unbalanced_weighted))

    ATE, ATE_w = cal_ATE(golds_treatment=t, logits_treatment=propensity, golds_outcome=y, normalized=True)
    UncorrectedEstimator_EY1_val, UncorrectedEstimator_EY0_val, ATE_original = ATE
    IPWEstimator_EY1_val, IPWEstimator_EY0_val, ATE_weighted = ATE_w
    print('LR: ATE_original: {}, ATE_weighted: {}'.format(ATE_original, ATE_weighted))

    n_unbalanced_feature = len(np.where(hidden_deviation > 0.1)[0])
    n_unbalanced_feature_w = len(np.where(hidden_deviation_w > 0.1)[0])

    print('LR: n_unbalanced_feature: {}, n_unbalanced_feature_w: {}'.format(n_unbalanced_feature, n_unbalanced_feature_w),
          flush=True)

    output_lr.write(
        '{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(n_user, n_nonuser,
                                           max_unbalanced_original, max_unbalanced_weighted,
                                           n_unbalanced_feature, n_unbalanced_feature_w,n_feature,
                                           UncorrectedEstimator_EY1_val, UncorrectedEstimator_EY0_val,
                                           ATE_original, IPWEstimator_EY1_val, IPWEstimator_EY0_val,
                                           ATE_weighted))


if __name__ == "__main__":
    main(args=parse_args())

