import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


def elbow_method(X, save_path):

    sse = []
    k_range = range(1, 10)

    for k in k_range:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(X)
        sse.append(model.inertia_)

    plt.figure()

    plt.plot(k_range, sse, marker="o")

    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("SSE")

    plt.title("Elbow Method")

    plt.savefig(save_path)

    plt.close()


def run_clustering(X):

    model = KMeans(n_clusters=3, random_state=42, n_init=10)

    clusters = model.fit_predict(X)

    return clusters


def plot_clusters(X, clusters, save_path):

    pca = PCA(n_components=2)

    X_pca = pca.fit_transform(X)

    plt.figure()

    plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=clusters
    )

    plt.title("KMeans Clusters (PCA Projection)")

    plt.savefig(save_path)

    plt.close()


def compute_silhouette(X, clusters):

    score = silhouette_score(X, clusters)

    return score