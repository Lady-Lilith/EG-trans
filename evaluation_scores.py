from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import numpy as np


def calculate_performace(num, y_pred, y_prob, y_test):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for index in range(num):
        if y_test[index] ==1:
            if y_test[index] == y_pred[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if y_test[index] == y_pred[index]:
                tn = tn + 1
            else:
                fp = fp + 1
    print("tp:",tp)
    print("tn:",tn)
    print("fp:",fp)
    print("fn:",fn)

    acc = round(float(tp + tn) / num, 5)
    tpr = round(tp / (tp + fn), 5)
    fpr = round(fp / (fp + tn), 5)

    try:
        precision = round(float(tp) / (tp + fp), 5)
        recall = round(float(tp) / (tp + fn), 5)
        f1_score = round((2 * precision * recall) / (precision + recall), 5)
        MCC = round((tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)), 5)
        sens = round(tp / (tp + fn), 5)
    except ZeroDivisionError:
        print("You can't divide by 0.")
        precision = recall = f1_score = sens = MCC = 100

    AUC = round(roc_auc_score(y_test, y_prob), 5)
    auprc = round(average_precision_score(y_test, y_prob), 5)

    return tp, fp,tpr,fpr, tn, fn, acc, precision, sens, f1_score, MCC, AUC,auprc