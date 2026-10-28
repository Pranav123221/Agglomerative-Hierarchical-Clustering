# Agglomerative Hierarchical Clustering: A Practical Guide to Unsupervised Learning

A comprehensive implementation of Agglomerative Hierarchical Clustering using Python, Scikit-learn, and SciPy, covering distance metrics, linkage criteria, dendrogram construction, cluster evaluation, and hierarchical cluster formation.

## 1. Project Overview

Agglomerative Hierarchical Clustering is an unsupervised machine learning algorithm that builds a hierarchy of clusters using a bottom-up approach.

Initially, every observation is treated as an individual cluster. The algorithm repeatedly merges the closest clusters according to a selected linkage criterion until the desired number of clusters is reached or the hierarchy is fully constructed.

Unlike K-Means, which partitions data around a predefined number of centroids, hierarchical clustering builds nested groups that can be examined at different levels of granularity.

### Objectives

- Understand the bottom-up hierarchical clustering mechanism.
- Implement Agglomerative Clustering using Scikit-learn.
- Analyze distance metrics and linkage strategies.
- Construct and interpret dendrograms using SciPy.
- Determine an appropriate number of clusters.
- Evaluate clustering quality using internal validation metrics.
- Understand the effects of feature scaling and cluster geometry.

### Tech Stack

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- SciPy

## 2. Repository Structure

```text
Agglomerative-Hierarchical-Clustering/
│
├── agglomerative_clustering.py
├── dendrogram_analysis.py
├── linkage_comparison.py
├── evaluation.py
├── requirements.txt
└── README.md
```

| File | Purpose |
|---|---|
| `agglomerative_clustering.py` | Main clustering implementation |
| `dendrogram_analysis.py` | Hierarchical tree visualization |
| `linkage_comparison.py` | Comparison of linkage strategies |
| `evaluation.py` | Evaluation using internal clustering metrics |
| `requirements.txt` | Project dependencies |
| `README.md` | Theory, implementation, and documentation |

## 3. What Is Hierarchical Clustering?

Hierarchical clustering organizes observations into a hierarchy represented by a tree-like structure.

There are two primary approaches:

**Agglomerative clustering — bottom-up**

- Begin with each observation as its own cluster.
- Calculate the distances between clusters.
- Merge the closest pair according to the linkage rule.
- Recalculate the relevant cluster distances.
- Continue until the stopping condition is reached.

**Divisive clustering — top-down**

- Begin with all observations in one cluster.
- Recursively divide clusters into smaller groups.

This project focuses on the agglomerative approach.

## 4. Mathematical Foundation

Let the dataset be:

\[
X=\{x_1,x_2,\ldots,x_n\}
\]

where each observation \(x_i\) is a feature vector in \(d\)-dimensional space.

Initially, each observation forms its own cluster:

\[
C_i=\{x_i\}, \qquad i=1,\ldots,n
\]

At every iteration, the algorithm selects two clusters with the smallest distance under the chosen linkage definition and merges them.

The process can be expressed as:

\[
(C_a,C_b)=
\arg\min_{C_i\neq C_j}D(C_i,C_j)
\]

The merged cluster becomes:

\[
C_{\text{new}}=C_a\cup C_b
\]

This process continues until the requested number of clusters is reached.

The exact merging sequence depends on the distance metric, linkage criterion, data representation, and tie-breaking behavior.

## 5. Distance Metrics

Distance metrics determine how dissimilarity between individual observations is measured.

### 5.1 Euclidean Distance

Euclidean distance measures the straight-line distance between two observations.

\[
d(x,y)=\sqrt{\sum_{i=1}^{d}(x_i-y_i)^2}
\]

It is commonly used with continuous numerical features.

### 5.2 Manhattan Distance

Manhattan distance calculates the sum of absolute coordinate differences.

\[
d(x,y)=\sum_{i=1}^{d}|x_i-y_i|
\]

It measures distance along coordinate axes and can behave differently from Euclidean distance when feature differences vary substantially.

### 5.3 Cosine Distance

Cosine distance measures the difference in orientation between two nonzero vectors.

\[
d_{\cos}(x,y)=
1-\frac{x\cdot y}{\|x\|\|y\|}
\]

It is useful for some high-dimensional representations, including text vectors and embeddings, where vector direction may be more informative than magnitude.

Cosine distance is not interchangeable with Euclidean distance, and its suitability depends on the representation and linkage strategy.

### Practical consideration

Distance calculations are sensitive to feature scales. If one feature ranges from 0 to 100,000 and another from 0 to 1, the larger-scale feature may dominate the distance.

Standardization is often useful before clustering continuous numerical features.

## 6. Linkage Criteria

A linkage criterion defines how the distance between two clusters is calculated.

### 6.1 Single Linkage

Single linkage uses the minimum distance between any pair of observations across two clusters.

\[
D_{\text{single}}(A,B)=
\min_{a\in A,b\in B}d(a,b)
\]

**Characteristics:**

- Can discover elongated or irregularly shaped clusters.
- Can connect clusters through chains of nearby points.
- Is susceptible to the chaining effect.

### 6.2 Complete Linkage

Complete linkage uses the maximum distance between observations across two clusters.

\[
D_{\text{complete}}(A,B)=
\max_{a\in A,b\in B}d(a,b)
\]

**Characteristics:**

- Tends to favor compact clusters.
- Considers the most distant pair across the clusters.
- Can be sensitive to outliers because of the maximum-distance calculation.

### 6.3 Average Linkage

Average linkage uses the average of all pairwise distances between observations in two clusters.

\[
D_{\text{average}}(A,B)=
\frac{1}{|A||B|}
\sum_{a\in A}\sum_{b\in B}d(a,b)
\]

**Characteristics:**

- Considers all cross-cluster pairwise distances.
- Often provides a compromise between single and complete linkage.
- Its behavior depends on the selected distance metric and dataset structure.

### 6.4 Ward Linkage

Ward linkage merges the pair of clusters that produces the smallest increase in within-cluster sum of squared errors.

For a cluster \(C\), define:

\[
W(C)=\sum_{x\in C}\|x-\mu_C\|_2^2
\]

where \(\mu_C\) is the cluster centroid.

The merge cost for clusters \(A\) and \(B\) is:

\[
\Delta W(A,B)=W(A\cup B)-W(A)-W(B)
\]

Ward linkage selects the merge with the smallest increase in this quantity.

**Characteristics:**

- Often produces compact, approximately spherical clusters.
- Is closely related to minimizing within-cluster variance.
- Requires Euclidean geometry in Scikit-learn's implementation.

Ward linkage does not simply use the maximum or minimum pairwise distance. It evaluates the increase in cluster variance caused by a merge.

### Linkage comparison

| Linkage | Cluster distance definition | Typical tendency |
|---|---|---|
| Single | Minimum pairwise distance | Can form elongated clusters |
| Complete | Maximum pairwise distance | Favors compact clusters |
| Average | Mean pairwise distance | Balances local and global distances |
| Ward | Increase in within-cluster SSE | Favors compact, low-variance clusters |

No linkage method is universally optimal. Compare them against the structure and objective of your dataset.

## 7. Dendrograms

A dendrogram is a tree diagram that represents the sequence of hierarchical cluster merges.

Each leaf represents an observation. Internal branches represent merges between clusters.

The vertical position of a merge represents its linkage distance or merge dissimilarity under the chosen linkage definition.

### How to interpret a dendrogram

- Leaves at the bottom represent individual observations.
- Branches indicate which observations or clusters merge.
- Lower merge heights indicate merges occurring at smaller linkage distances.
- Larger vertical jumps between successive merges may suggest candidate cut levels.
- Drawing a horizontal line across the tree produces a flat clustering.

A dendrogram does not automatically determine the correct number of clusters. The cut level should be considered alongside validation metrics, domain knowledge, and practical usefulness.

## 8. Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Agglomerative-Hierarchical-Clustering.git

cd Agglomerative-Hierarchical-Clustering
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### requirements.txt

```text
numpy
pandas
matplotlib
scikit-learn
scipy
```

## 9. Practical Implementation: Agglomerative Clustering

This example generates a synthetic dataset with four underlying groups, scales the features, fits Agglomerative Clustering, and evaluates the resulting labels.

The generated ground-truth labels are intentionally not used to train the model. The purpose is to demonstrate an unsupervised learning workflow.

### agglomerative_clustering.py

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)

# 1. Generate synthetic data
X, _ = make_blobs(
    n_samples=300,
    centers=4,
    n_features=2,
    cluster_std=1.2,
    random_state=42
)

df = pd.DataFrame(
    X,
    columns=["Feature_1", "Feature_2"]
)

# 2. Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# 3. Initialize the model
model = AgglomerativeClustering(
    n_clusters=4,
    metric="euclidean",
    linkage="ward"
)

# 4. Fit and predict cluster assignments
labels = model.fit_predict(X_scaled)

df["Cluster"] = labels

# 5. Evaluate clustering
silhouette = silhouette_score(X_scaled, labels)
calinski = calinski_harabasz_score(X_scaled, labels)
davies_bouldin = davies_bouldin_score(X_scaled, labels)

print("Silhouette Score:", silhouette)
print("Calinski-Harabasz Index:", calinski)
print("Davies-Bouldin Index:", davies_bouldin)

# 6. Inspect cluster sizes
print("\nCluster sizes:")
print(df["Cluster"].value_counts().sort_index())

# 7. Visualize clusters
plt.figure(figsize=(9, 6))

plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=labels,
    cmap="viridis",
    s=40,
    alpha=0.8
)

plt.xlabel("Feature 1 (standardized)")
plt.ylabel("Feature 2 (standardized)")
plt.title("Agglomerative Hierarchical Clustering")
plt.tight_layout()
plt.show()
```

### Implementation explanation

1. `make_blobs()` generates numerical observations with underlying groups for demonstration.
2. `StandardScaler` standardizes the two features.
3. `AgglomerativeClustering` is configured to form four clusters using Ward linkage.
4. `fit_predict()` builds the hierarchy needed for the requested partition and returns cluster assignments.
5. Three internal metrics evaluate cohesion and separation.
6. The scatter plot visualizes the resulting cluster assignments.

The synthetic generator's four groups are known for demonstration, but the clustering algorithm itself does not use those labels.

## 10. Dendrogram Analysis with SciPy

Scikit-learn's `AgglomerativeClustering` returns cluster labels but does not directly provide the complete linkage matrix needed by SciPy's standard dendrogram function.

SciPy's `linkage()` can construct that matrix directly from the observations.

### dendrogram_analysis.py

```python
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram

# 1. Generate data
X, _ = make_blobs(
    n_samples=30,
    centers=4,
    n_features=2,
    random_state=42
)

# 2. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Construct hierarchical linkage matrix
Z = linkage(
    X_scaled,
    method="ward",
    metric="euclidean"
)

# 4. Plot dendrogram
plt.figure(figsize=(12, 6))

dendrogram(
    Z,
    leaf_rotation=90,
    leaf_font_size=8
)

plt.title("Agglomerative Hierarchical Clustering Dendrogram")
plt.xlabel("Observation Index")
plt.ylabel("Ward Linkage Distance")
plt.tight_layout()
plt.show()
```

### Understanding the linkage matrix

The matrix `Z` contains one row for each merge.

For \(n\) observations, it has \(n-1\) rows when the complete hierarchy is constructed.

Each row contains four values:

1. Index of the first merged cluster or observation.
2. Index of the second merged cluster or observation.
3. Distance or merge height associated with the merge.
4. Number of original observations contained in the newly formed cluster.

For this example, the dendrogram is based on the 30 standardized observations.

**Important:** SciPy's Ward linkage heights are related to the increase in within-cluster sum of squared errors, but they are not simply identical to the raw SSE increase. Use the dendrogram to interpret merge heights, not to assume every displayed height is a raw Euclidean distance between centroids.

## 11. Comparing Linkage Methods

Different linkage criteria can produce different cluster structures from the same data.

### linkage_comparison.py

```python
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

# Generate a shared dataset
X, _ = make_blobs(
    n_samples=300,
    centers=4,
    n_features=2,
    cluster_std=1.2,
    random_state=42
)

# Scale once for this comparison
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

methods = [
    ("single", "Single Linkage"),
    ("complete", "Complete Linkage"),
    ("average", "Average Linkage"),
    ("ward", "Ward Linkage")
]

fig, axes = plt.subplots(
    2, 2,
    figsize=(12, 9)
)

for ax, (method, title) in zip(axes.flat, methods):
    model = AgglomerativeClustering(
        n_clusters=4,
        metric="euclidean",
        linkage=method
    )

    labels = model.fit_predict(X_scaled)

    ax.scatter(
        X_scaled[:, 0],
        X_scaled[:, 1],
        c=labels,
        cmap="viridis",
        s=25,
        alpha=0.8
    )

    ax.set_title(title)
    ax.set_xlabel("Feature 1 (standardized)")
    ax.set_ylabel("Feature 2 (standardized)")

plt.tight_layout()
plt.show()
```

This experiment holds the dataset, feature scaling, number of clusters, and distance metric constant while changing the linkage strategy.

Observe how the merge rule influences the resulting partition. The outputs should be interpreted rather than assumed to be identical across methods.

## 12. Cluster Evaluation

Because clustering is unsupervised, ground-truth labels may not be available. Internal validation metrics assess aspects of cluster cohesion and separation without requiring known labels.

### 12.1 Silhouette Score

The Silhouette Score compares the mean distance of an observation to its own cluster with its mean distance to the nearest alternative cluster.

For observation \(i\):

\[
s(i)=\frac{b(i)-a(i)}
{\max(a(i),b(i))}
\]

Where:

- \(a(i)\) is the mean distance to other observations in the same cluster.
- \(b(i)\) is the smallest mean distance to observations in another cluster.

Interpretation:

- Close to 1: The observation is relatively well separated from other clusters.
- Close to 0: The observation lies near a cluster boundary.
- Below 0: The observation may be closer to another cluster.

Higher overall scores are generally preferred when comparing comparable candidate clusterings.

### 12.2 Calinski-Harabasz Index

The Calinski-Harabasz Index compares between-cluster dispersion against within-cluster dispersion.

Higher values generally indicate more distinct, compact groups under this criterion.

### 12.3 Davies-Bouldin Index

The Davies-Bouldin Index measures similarity between clusters based on their dispersion and separation.

Lower values are generally preferred.

### evaluation.py

```python
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)

X, _ = make_blobs(
    n_samples=300,
    centers=4,
    n_features=2,
    random_state=42
)

X_scaled = StandardScaler().fit_transform(X)

for k in range(2, 8):
    model = AgglomerativeClustering(
        n_clusters=k,
        linkage="ward",
        metric="euclidean"
    )

    labels = model.fit_predict(X_scaled)

    print(f"\nNumber of clusters: {k}")
    print("Silhouette Score:",
          silhouette_score(X_scaled, labels))
    print("Calinski-Harabasz Index:",
          calinski_harabasz_score(X_scaled, labels))
    print("Davies-Bouldin Index:",
          davies_bouldin_score(X_scaled, labels))
```

This script compares candidate cluster counts from 2 to 7.

Metric values should be used as diagnostic evidence, not as an automatic guarantee of the best real-world clustering. The metrics favor certain cluster geometries and may disagree.

## 13. Choosing the Number of Clusters

The number of clusters can be selected using several complementary methods.

**Dendrogram cut:** Identify a potential horizontal cut that separates groups across a relatively large gap in merge heights.

**Silhouette analysis:** Compare the average Silhouette Score for different cluster counts.

**Calinski-Harabasz and Davies-Bouldin indices:** Evaluate compactness and separation from different perspectives.

**Domain knowledge:** Consider whether the resulting groups are useful and interpretable for the application.

A robust choice should balance internal validation, stability, computational cost, and the purpose of the analysis.

## 14. Important Hyperparameters

| Parameter | Description |
|---|---|
| `n_clusters` | Number of flat clusters to produce |
| `metric` | Distance metric used for pairwise distances |
| `linkage` | Cluster-to-cluster merge criterion |
| `compute_full_tree` | Controls whether to compute the full tree or stop once the requested clustering is obtained, subject to implementation details |
| `distance_threshold` | Distance threshold used to determine when merging stops |
| `compute_distances` | Requests storage of distances between merged clusters |

In Scikit-learn, `n_clusters` and `distance_threshold` must not both be set to non-`None` values.

For a dendrogram workflow, the full hierarchy is required. SciPy's `linkage()` directly constructs the full hierarchy for the supplied observations.

## 15. Computational Complexity

Agglomerative clustering can be computationally expensive for large datasets because it must repeatedly identify and process cluster merges.

The precise runtime and memory requirements depend on the implementation, linkage criterion, input structure, and optimizations.

A common practical concern is that standard hierarchical clustering methods can require quadratic memory in the number of observations, particularly when pairwise distance information is stored.

For very large datasets, alternatives such as MiniBatch K-Means, sampling-based exploration, or scalable hierarchical methods may be more suitable.

## 16. Advantages

- Does not require centroid-based cluster representations.
- Provides a hierarchy of nested groups.
- Supports multiple linkage strategies.
- Enables dendrogram-based interpretation.
- Does not require ground-truth labels.
- Can be useful for discovering relationships between observations at multiple levels of granularity.

## 17. Limitations

- Can become computationally expensive on large datasets.
- Early merge decisions generally cannot be undone.
- Is sensitive to feature scaling and distance definitions.
- Different linkage methods may produce substantially different results.
- Is not inherently immune to outliers.
- Does not automatically determine the semantically correct number of clusters.
- Cluster IDs are arbitrary and do not indicate a ranking.
- Dendrograms become difficult to interpret when the number of observations is large.

## 18. Agglomerative Clustering vs. K-Means

| Characteristic | Agglomerative Clustering | K-Means |
|---|---|---|
| Learning type | Unsupervised | Unsupervised |
| Core mechanism | Repeated cluster merging | Alternating assignment and centroid updates |
| Number of clusters | Can use a distance threshold or a specified count | Requires a specified count |
| Hierarchical structure | Yes | No |
| Centroids | Not required for the general algorithm | Central to the algorithm |
| Main configuration | Linkage and distance metric | Centroid initialization and cluster count |
| Scalability | Often expensive for large datasets | Often more scalable |
| Dendrogram | Available through hierarchical linkage tools | Not applicable to standard K-Means |
| Typical use | Exploring nested cluster relationships | Efficient partitioning around centroids |

## 19. Best Practices

- Standardize numerical features when scales differ substantially.
- Compare multiple linkage strategies.
- Use a consistent dataset when comparing methods.
- Inspect cluster sizes and scatter plots where dimensionality permits.
- Use internal evaluation metrics alongside domain knowledge.
- Examine dendrogram merge heights rather than choosing a cluster count arbitrarily.
- Use the same fitted preprocessing transformation for subsequent inference.
- Avoid assuming that a higher metric score automatically means a more meaningful business result.
- For high-dimensional data, consider dimensionality reduction for visualization, while evaluating the effect of that transformation on cluster structure.
- Record dependency versions and random seeds where applicable for reproducibility.

## 20. Key Takeaways

Agglomerative Hierarchical Clustering constructs a hierarchy by repeatedly merging clusters according to a selected linkage criterion.

The central concepts are:

1. **Bottom-up merging:** Start with individual observations and progressively combine them.
2. **Distance metrics:** Quantify dissimilarity between observations.
3. **Linkage criteria:** Define how cluster-to-cluster distance is calculated.
4. **Dendrograms:** Visualize the sequence and heights of hierarchical merges.
5. **Cluster evaluation:** Use internal metrics to examine cohesion and separation.
6. **Feature scaling:** Ensure numerical ranges do not unintentionally dominate distance calculations.
7. **Model selection:** Combine dendrogram interpretation, validation metrics, and application-specific requirements.

The key lesson is that hierarchical clustering is not merely a way to assign cluster labels. It is a framework for examining how data points relate to one another across multiple levels of grouping.

## References

- [Scikit-learn: AgglomerativeClustering API](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html)
- [Scikit-learn: Clustering guide](https://scikit-learn.org/stable/modules/clustering.html)
- [SciPy: Hierarchical clustering](https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html)
- [SciPy: linkage function](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html)
- [SciPy: dendrogram function](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.dendrogram.html)

---

### Project Tags

`Machine Learning` · `Unsupervised Learning` · `Agglomerative Clustering` · `Hierarchical Clustering` · `Dendrogram` · `Linkage Criteria` · `Distance Metrics` · `Cluster Evaluation` · `Python` · `Scikit-Learn` · `SciPy` · `Data Science`
