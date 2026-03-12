import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve


def plot_confusion_matrix(y_true, y_pred, name, save_path):

    cm = confusion_matrix(y_true, y_pred)

    plt.figure()

    sns.heatmap(
        cm,
        annot=True,
        fmt="d"
    )

    plt.title(f"Confusion Matrix - {name}")

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.savefig(save_path)

    plt.close()


def plot_roc_curve(y_true, y_prob, name, save_path):

    fpr, tpr, _ = roc_curve(y_true, y_prob)

    plt.figure()

    plt.plot(fpr, tpr)

    plt.plot([0, 1], [0, 1], "--")

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.title(f"ROC Curve - {name}")

    plt.savefig(save_path)

    plt.close()