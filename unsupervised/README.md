# Oil & Gas Production Dataset (1967-1999)
## Unsupervised Learning Analysis

### Overview
This project applies unsupervised learning techniques to analyze oil and gas production data from 1967-1999. The analysis includes exploratory data analysis, dimensionality reduction, and clustering with four different algorithms.

---

### Dataset
- **Source:** `oil-and-gas-summary-production-data-1967-1999-1.csv`
- **Size:** 30,053 rows × 20 columns
- **Time Period:** 1967-1999
- **Features:** Production metrics, well counts, county information, geological formations

#### Missing Values
| Column | Missing |
|--------|---------|
| Purchaser Codes | 11,798 |
| Field | 1,281 |
| Producing Formation | 660 |
| Town | 657 |
| Self-use Well | 619 |
| County | 31 |

---

### Methodology

#### 1. Feature Engineering
Created 6 new features:
- `Total_Wells` - Sum of all well types
- `Is_Gas_Producer` - Binary flag for gas production
- `Is_Oil_Producer` - Binary flag for oil production
- `Production_Total` - Combined oil + gas production
- `Water_Ratio` - Water to total production ratio
- `Oil_per_Well` / `Gas_per_Well` - Per-well productivity

**Final Feature Matrix:** 30,053 samples × 21 features

#### 2. Dimensionality Reduction
- **PCA (2 components):** 34.9% variance explained
- **PCA (85% variance):** Requires 12 components
- **t-SNE:** KL divergence = 0.7239 (10,000 samples)

The low PCA variance with 2 components indicates the data is high-dimensional with many independent factors.

---

### Clustering Results

#### K-Means (k=2) — **WINNER**
| Metric | Value |
|--------|-------|
| Silhouette Score | **0.8484** |
| Davies-Bouldin Index | 1.3183 |
| Calinski-Harabasz Score | 3042.6 |

**Cluster Profiles:**
| Cluster | Active Oil Wells | Active Gas Wells | Oil (bbl) | Gas (Mcf) | Water (bbl) | Total Wells |
|---------|-----------------|-----------------|-----------|-----------|-------------|-------------|
| 0 (Small) | 3.52 | 3.67 | 389 | 17,372 | 729 | 9.53 |
| 1 (Large) | 212.78 | 11.83 | 58,721 | 119,479 | 761,979 | 360 |

**Interpretation:** Clear separation between small/independent producers and large industrial operations.

#### Agglomerative Clustering (k=3)
| Metric | Value |
|--------|-------|
| Silhouette Score | 0.3288 |
| Davies-Bouldin Index | 1.6267 |
| Calinski-Harabasz Score | 2839.6 |

#### DBSCAN (eps=1.5, min_samples=20)
| Metric | Value |
|--------|-------|
| Silhouette Score | 0.2447 |
| Clusters Found | 6 |
| Noise Points | 1,898 (6.3%) |

#### Gaussian Mixture Model (n=8)
| Metric | Value |
|--------|-------|
| Silhouette Score | 0.1982 |
| Davies-Bouldin Index | 2.3633 |
| Calinski-Harabasz Score | 1454.1 |
| BIC | -3,616,473 |

---

### Algorithm Comparison Summary

| Algorithm | Silhouette ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ |
|-----------|-------------|------------------|---------------------|
| **KMeans (k=2)** | **0.8484** | 1.3183 | 3042.6 |
| Agglomerative (k=3) | 0.3288 | 1.6267 | 2839.6 |
| DBSCAN (eps=1.5) | 0.2447 | 1.6482 | 2899.1 |
| GMM (n=8) | 0.1982 | 2.3633 | 1454.1 |

---

### Final Verdict

**K-Means with k=2** is the best performing algorithm:
- Highest Silhouette Score (0.8484) — indicates strong cluster separation
- Lowest Davies-Bouldin Index (1.3183) — indicates compact, well-separated clusters
- High Calinski-Harabasz Score (3042.6) — indicates dense clusters

**Business Insight:** The data naturally splits into two distinct groups:
1. **Small Producers** — Fewer wells, lower production volumes
2. **Large Producers** — Industrial-scale operations with 60x more wells and 150x more oil production

---

### Inline Visualizations (displayed when running notebook)

All plots render inline in the notebook — no image files are saved to disk.

| Cell | Plot |
|------|------|
| EDA | 3×4 grid of feature distributions (log scale for large values) |
| EDA | Correlation heatmap of numeric features |
| EDA | Oil and gas production trends over years (dual line chart) |
| EDA | Top 10 counties by oil and gas production (horizontal bar charts) |
| PCA | Cumulative explained variance curve with 85% threshold line |
| K-Means | Elbow curve, Silhouette vs k, Davies-Bouldin vs k |
| K-Means | PCA and t-SNE scatter plots colored by cluster |
| K-Means | Normalized cluster profile heatmap |
| Agglomerative | PCA and t-SNE scatter plots colored by cluster |
| DBSCAN | PCA and t-SNE scatter plots colored by cluster |
| GMM | BIC & AIC curves + Silhouette vs components |
| GMM | PCA and t-SNE scatter plots colored by cluster |
| Comparison | 3 bar charts comparing Silhouette, DB, and CH scores |

---

### Requirements
```
numpy
pandas
matplotlib
seaborn
scikit-learn
```

### Usage
```bash
jupyter notebook unsupervised.ipynb
```
Run all cells sequentially (Cell → Run All).
