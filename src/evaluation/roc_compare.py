import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve


def plot_roc_comparison(y_true, prob_dict, save_path):

    plt.figure()

    for name, prob in prob_dict.items():

        fpr, tpr, _ = roc_curve(y_true, prob)

        plt.plot(fpr, tpr, label=name)

    plt.plot([0, 1], [0, 1], "--")

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.title("ROC Comparison of Models")

    plt.legend()

    plt.savefig(save_path)

    plt.close()