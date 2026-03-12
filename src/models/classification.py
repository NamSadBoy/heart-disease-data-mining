from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


def get_models():

    models = {}

    models["LogisticRegression"] = LogisticRegression(max_iter=1000)

    models["RandomForest"] = RandomForestClassifier()

    models["SVM"] = SVC(probability=True)

    models["DecisionTree"] = DecisionTreeClassifier()

    models["KNN"] = KNeighborsClassifier()

    return models