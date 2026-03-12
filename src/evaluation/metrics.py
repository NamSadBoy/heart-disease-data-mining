from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score


def evaluate_model(y_true, y_pred, y_prob):

    acc = accuracy_score(y_true, y_pred)

    f1 = f1_score(y_true, y_pred)

    roc = roc_auc_score(y_true, y_prob)

    return acc, f1, roc