import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from mlxtend.frequent_patterns import apriori, association_rules

from src.mining.clustering import (
    run_clustering,
    elbow_method,
    plot_clusters,
    compute_silhouette
)

from src.evaluation.metrics import evaluate_model

from src.evaluation.visualization import (
    plot_confusion_matrix,
    plot_roc_curve
)

from src.evaluation.roc_compare import plot_roc_comparison

from src.models.semi_supervised import semi_supervised_experiment


def main():

    print("Loading data...")

    df = pd.read_csv("data/raw/heart.csv")

    df.columns = df.columns.str.strip()

    # rename label
    if "num" in df.columns:
        df = df.rename(columns={"num": "target"})

    # convert to binary classification
    df["target"] = (df["target"] > 0).astype(int)

    print("Dataset shape:", df.shape)

    
    # =========================
    # PREPROCESSING
    # =========================

    print("Preprocessing data...")

    # tách feature và label
    X = df.drop("target", axis=1)
    y = df["target"]

    # encode categorical
    X = pd.get_dummies(X)

    # xử lý missing values
    from sklearn.impute import SimpleImputer

    imputer = SimpleImputer(strategy="median")

    X = imputer.fit_transform(X)

    # scaling
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # train test split
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42
    )


    # =========================
    # ASSOCIATION RULES
    # =========================

    print("Running Association Rules...")

    basket = df.copy()

    # xử lý missing values
    basket = basket.fillna(0)

    # convert categorical -> one-hot
    basket = pd.get_dummies(basket)

    # convert sang bool cho Apriori
    basket = basket.astype(bool)

    freq_items = apriori(
        basket,
        min_support=0.1,
        use_colnames=True
    )

    rules = association_rules(
        freq_items,
        metric="lift",
        min_threshold=1
    )

    os.makedirs("outputs/tables", exist_ok=True)

    rules.to_csv(
        "outputs/tables/association_rules.csv",
        index=False
    )


    # =========================
    # CLUSTERING
    # =========================

    print("Running Clustering...")

    os.makedirs("outputs/figures", exist_ok=True)

    clusters = run_clustering(X_scaled)

    plot_clusters(
        X_scaled,
        clusters,
        "outputs/figures/clusters.png"
    )

    elbow_method(
        X_scaled,
        "outputs/figures/elbow.png"
    )

    sil_score = compute_silhouette(X_scaled, clusters)

    print("Silhouette score:", sil_score)


    # =========================
    # CLASSIFICATION MODELS
    # =========================

    print("Training Classification Models...")

    models = {

        "LogisticRegression":
            LogisticRegression(max_iter=1000),

        "RandomForest":
            RandomForestClassifier(),

        "SVM":
            SVC(probability=True),

        "DecisionTree":
            DecisionTreeClassifier(),

        "KNN":
            KNeighborsClassifier()

    }

    results = []

    prob_dict = {}


    for name, model in models.items():

        print("Training", name)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        y_prob = model.predict_proba(X_test)[:, 1]

        acc, f1, roc = evaluate_model(
            y_test,
            y_pred,
            y_prob
        )

        results.append([
            name,
            acc,
            f1,
            roc
        ])

        prob_dict[name] = y_prob

        plot_confusion_matrix(
            y_test,
            y_pred,
            name,
            f"outputs/figures/cm_{name}.png"
        )

        plot_roc_curve(
            y_test,
            y_prob,
            name,
            f"outputs/figures/roc_{name}.png"
        )


    results_df = pd.DataFrame(
        results,
        columns=["Model", "Accuracy", "F1", "ROC_AUC"]
    )

    results_df.to_csv(
        "outputs/tables/classification_results.csv",
        index=False
    )


    # =========================
    # ROC COMPARISON
    # =========================

    plot_roc_comparison(
        y_test,
        prob_dict,
        "outputs/figures/roc_comparison.png"
    )


    # =========================
    # SEMI SUPERVISED
    # =========================

    print("Running Semi-Supervised...")

    semi_results = semi_supervised_experiment(
        X_train,
        y_train,
        X_test,
        y_test
    )

    semi_df = pd.DataFrame(
        semi_results,
        columns=["Label_Percent", "Accuracy"]
    )

    semi_df.to_csv(
        "outputs/tables/semi_supervised_results.csv",
        index=False
    )


    print("Pipeline completed successfully!")


if __name__ == "__main__":
    main()