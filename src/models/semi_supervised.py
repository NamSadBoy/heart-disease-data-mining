import numpy as np
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import accuracy_score


def semi_supervised_experiment(X_train, y_train, X_test, y_test):

    percents = [0.05, 0.1, 0.2]

    results = []

    for p in percents:

        mask = np.random.rand(len(y_train)) < p

        y_semi = y_train.copy()

        y_semi[~mask] = -1

        model = LabelSpreading()

        model.fit(X_train, y_semi)

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        results.append((p, acc))

    return results