import numpy as np

data = np.array([
    [1, 1],
    [4, 1],
    [6, 1],
    [1, 2],
    [2, 3],
    [5, 3],
    [2, 5],
    [3, 5],
    [2, 6],
    [3, 8]
])

K = 3
threshold = 0.1

centroids = np.array([
    data[0],
    data[2],
    data[1]
], dtype=float)

def euclidean(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

for iteration in range(100):
    clusters = [[] for _ in range(K)]
    labels = []

    for point in data:
        dists = [euclidean(point, c) for c in centroids]
        idx = np.argmin(dists)
        labels.append(idx)
        clusters[idx].append(point)

    new_centroids = []
    for i in range(K):
        if clusters[i]:
            new_centroids.append(np.mean(clusters[i], axis=0))
        else:
            new_centroids.append(centroids[i])

    new_centroids = np.array(new_centroids)
    delta = sum(euclidean(centroids[i], new_centroids[i]) for i in range(K))

    if delta < threshold:
        centroids = new_centroids
        break

    centroids = new_centroids

print("\n=== HASIL AKHIR ===")
for i, p in enumerate(data):
    print(f"Data {i+1} {tuple(p)} -> Cluster K{labels[i]+1}")

print("\nCentroid akhir:")
for i, c in enumerate(centroids):
    print(f"K{i+1} = ({c[0]:.4f}, {c[1]:.4f})")
