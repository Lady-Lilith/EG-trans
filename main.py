import numpy as np
from matplotlib import pyplot as plt
from numpy import interp
from sklearn.neural_network import MLPClassifier
# from xgboost import XGBClassifier
import torch.nn as nn
from param import parameter_parser
import load_data
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from model import mymodel
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import evaluation_scores
# from figure import plot_auc_curves,plot_prc_curves
import time
import random
import torch
# Set random seeds
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# Use the seed value of your choice
set_seed(42)

def CDA(n_fold=5):
    args = parameter_parser()
    dataset, cd_pairs,cc_edge_index,dd_edge_index,one_index,cd_edge_index,dis,cis= load_data.load_dataset(args)

    # Initialize KFold
    kf = KFold(n_splits=n_fold, shuffle=True,random_state=42)
    mean_fpr = np.linspace(0, 1, 100)
    mean_recall = np.linspace(0, 1, 100)

    model = mymodel(args)
    i = 0
    ave_acc = 0
    ave_prec = 0
    ave_sens = 0
    ave_f1_score = 0
    ave_mcc = 0
    ave_auc = 0
    ave_auprc = 0
    aucs=[]
    precisions_list = []
    recalls_list = []
    average_precisions = []
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    tprs_list = []
    fprs_list = []

    for train_index, test_index in kf.split(cd_pairs):
        c_dmatix, train_cd_pairs, test_cd_pairs,cd_data = load_data.C_Dmatix(cd_pairs, train_index, test_index,)
        dataset['c_d'] = c_dmatix
        score, cir_fea, dis_fea = load_data.feature_representation(model, args, dataset)  # 进入评估模式
        train_dataset = load_data.new_dataset(cir_fea, dis_fea, train_cd_pairs)
        test_dataset = load_data.new_dataset(cir_fea, dis_fea, test_cd_pairs)
        X_train, y_train = train_dataset[:, :-2], train_dataset[:, -2:]
        X_test, y_test = test_dataset[:, :-2], test_dataset[:, -2:]
        print(X_train.shape, X_test.shape)
        clf = RandomForestClassifier(n_estimators=200, n_jobs=11, max_depth=20)
        clf.fit(X_train, y_train[:,0])
        y_pred = clf.predict(X_test)
            # y_pred = y_pred[:, 0]
        y_prob = clf.predict_proba(X_test)
            # y_prob = y_prob[1][:, 0]
        y_prob = y_prob[:, 1]
        tp, fp, tpr,fpr,tn, fn, acc, prec, sens, f1_score, MCC, AUC, AUPRC = evaluation_scores.calculate_performace(
            len(y_pred), y_pred, y_prob, y_test[:, 0])
        print('RF: \n  Acc = \t', acc, '\n  prec = \t', prec, '\n  sens = \t', sens, '\n  f1_score = \t', f1_score,
                  '\n  MCC = \t', MCC, '\n  AUC = \t', AUC, '\n  AUPRC = \t', AUPRC)
        # f.write('RF: \t  tp = \t' + str(tp) + '\t fp = \t' + str(fp) + '\t tn = \t' + str(tn) + '\t fn = \t' + str(
        #         fn) + '\t  Acc = \t' + str(acc) + '\t  prec = \t' + str(prec) + '\t  sens = \t' + str(
        #         sens) + '\t  f1_score = \t' + str(f1_score) + '\t  MCC = \t' + str(MCC) + '\t  AUC = \t' + str(
        #         AUC) + '\t  AUPRC = \t' + str(AUPRC) + '\n')

        ave_acc += acc
        ave_prec += prec
        ave_sens += sens
        ave_f1_score += f1_score
        ave_mcc += MCC
        ave_auc += AUC
        ave_auprc += AUPRC

        fpr, tpr, _ = roc_curve(y_test[:, 0], y_prob)
        if len(fpr) > 0 and len(tpr) > 0:
            tprs_list.append(interp(mean_fpr, fpr, tpr))
            tprs_list[-1][0] = 0.0
        precision, recall, _ = precision_recall_curve(y_test[:, 0], y_prob)
        if len(precision) > 0 and len(recall) > 0:
            precisions_list.append(np.interp(mean_recall, recall[::-1], precision[::-1]))
            average_precisions.append(average_precision_score(y_test[:, 0], y_prob))

        # 计算AUC并保存
        auc_score = auc(fpr, tpr)
        aucs.append(auc_score)


    ave_acc /= n_fold
    ave_prec /= n_fold
    ave_sens /= n_fold
    ave_f1_score /= n_fold
    ave_mcc /= n_fold
    ave_auc /= n_fold
    ave_auprc /= n_fold

    print('Final: \t tp = \t' + str(tp) + '\t fp = \t' + str(fp) + '\t tn = \t' + str(tn) + '\t fn = \t' + str(
        fn) + '\t,  Acc = \t' + str(round(ave_acc, 4)) +
          '\n' + '\t prec = \t' + str(round(ave_prec, 4)) +
          '\n' + '\t sens = \t' + str(round(ave_sens, 4)) + '\t f1_score = \t' + str(
        round(ave_f1_score, 4)) + '\t MCC = \t' + str(round(ave_mcc, 4)) +
          '\n' + '\t AUC = \t' + str(round(ave_auc, 4)) + '\t AUPRC = \t' + str(round(ave_auprc, 4)) + '\n')

    # ROC
    for i, tpr in enumerate(tprs_list):
        axs[0].plot(mean_fpr, tpr, alpha=0.5, linestyle='-', linewidth=1, label=f'Fold {i + 1} (AUC = {aucs[i]:.5f})')
    if tprs_list:
        mean_tpr = np.mean(tprs_list, axis=0)
        # mean_auc = auc(mean_fpr, mean_tpr)
        mean_auc = np.mean(aucs)
        axs[0].plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (AUC = {mean_auc:.4f})', linewidth=2.5,
                    linestyle='-')
    axs[0].plot([0, 1], [0, 1], linestyle='--', color='gray', alpha=0.7)
    axs[0].set_title('ROC Curve', fontsize=14, fontweight='bold')
    axs[0].set_xlabel('False Positive Rate (FPR)', fontsize=12)
    axs[0].set_ylabel('True Positive Rate (TPR)', fontsize=12)
    axs[0].legend(loc='lower right')
    axs[0].grid(True, linestyle='--', alpha=0.5)

    # Plot PR curves
    for i, precision in enumerate(precisions_list):
        axs[1].plot(mean_recall, precision, alpha=0.5, linestyle='-', linewidth=1,
                    label=f'Fold {i + 1} (AP = {average_precisions[i]:.5f})')
    if precisions_list:
        mean_precision = np.mean(precisions_list, axis=0)
        mean_ap = np.mean(average_precisions)
        axs[1].plot(mean_recall, mean_precision, color='b', label=f'Mean PR (AP = {mean_ap:.4f})', linewidth=2.5,
                    linestyle='-')

    # Add a baseline horizontal line for PR curve
    # axs[1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Baseline')

    axs[1].set_title('PR Curve', fontsize=14, fontweight='bold')
    axs[1].set_xlabel('Recall', fontsize=12)
    axs[1].set_ylabel('Precision', fontsize=12)
    axs[1].legend(loc='lower left')
    axs[1].grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    n_fold = 5
    for i in range(1):
        CDA(n_fold)
