# ============================================================
#  Customer Segmentation — K-Means Clustering
#  Python 3.8+  |  pip install pandas scikit-learn matplotlib seaborn
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# STEP 1 — Sample data உருவாக்குவோம்
#          (உங்களுக்கு real data இருந்தால்
#           df = pd.read_csv("your_file.csv") மாத்துங்க)
# ─────────────────────────────────────────────
np.random.seed(42)
N = 500

def make_segment(n, freq_mu, freq_sd, aov_mu, aov_sd,
                 recency_mu, recency_sd, age_mu, age_sd):
    return pd.DataFrame({
        "purchase_frequency": np.random.normal(freq_mu,    freq_sd,    n).clip(0),
        "avg_order_value":    np.random.normal(aov_mu,     aov_sd,     n).clip(10),
        "recency_days":       np.random.normal(recency_mu, recency_sd, n).clip(1),
        "age":                np.random.normal(age_mu,     age_sd,     n).clip(18, 75).astype(int),
        "total_orders":       np.random.randint(5, 80, n),
    })

df = pd.concat([
    make_segment(110, 8.2, 1.2, 480, 60,  15,  8,  38, 6),   # High-value loyalists
    make_segment(150, 2.1, 0.8, 210, 40,  45, 15,  34, 7),   # Occasional spenders
    make_segment(140, 4.8, 1.0,  85, 20,  30, 12,  28, 5),   # Bargain hunters
    make_segment(100, 0.6, 0.4, 175, 50, 120, 30,  46, 8),   # At-risk churners
], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

print("✅ Data உருவாக்கப்பட்டது:", df.shape)
print(df.describe().round(1))

# ─────────────────────────────────────────────
# STEP 2 — Features தேர்வு செய்து Scale பண்ணுவோம்
# ─────────────────────────────────────────────
features = ["purchase_frequency", "avg_order_value", "recency_days", "total_orders"]
X = df[features].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("\n✅ Features scaled செய்யப்பட்டது")

# ─────────────────────────────────────────────
# STEP 3 — Optimal K கண்டுபிடிக்கலாம் (Elbow + Silhouette)
# ─────────────────────────────────────────────
inertias, sil_scores = [], []
K_range = range(2, 9)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, km.labels_))

best_k = K_range[np.argmax(sil_scores)]
print(f"\n✅ Best K = {best_k}  (Silhouette score: {max(sil_scores):.3f})")

# ─────────────────────────────────────────────
# STEP 4 — Final K-Means Model
# ─────────────────────────────────────────────
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_scaled)

# Cluster-க்கு பெயர் வையுங்க (profile பார்த்து மாத்துங்க)
cluster_names = {
    0: "High-value loyalists",
    1: "Occasional spenders",
    2: "Bargain hunters",
    3: "At-risk churners",
}
df["segment"] = df["cluster"].map(cluster_names)

# ─────────────────────────────────────────────
# STEP 5 — Segment Profile
# ─────────────────────────────────────────────
print("\n📊 Segment Profile:\n")
profile = df.groupby("segment")[features + ["age"]].mean().round(1)
profile["count"] = df["segment"].value_counts()
profile["pct"]   = (profile["count"] / len(df) * 100).round(1)
print(profile.to_string())

# ─────────────────────────────────────────────
# STEP 6 — Visualizations
# ─────────────────────────────────────────────
COLORS = {
    "High-value loyalists": "#534AB7",
    "Occasional spenders":  "#0F6E56",
    "Bargain hunters":      "#D85A30",
    "At-risk churners":     "#888780",
}

sns.set_style("whitegrid")
fig = plt.figure(figsize=(18, 14))
fig.suptitle("Customer Segmentation Dashboard", fontsize=18, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# --- Plot 1: Elbow Curve ---
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(K_range, inertias, "o-", color="#534AB7", linewidth=2)
ax1.axvline(4, color="#D85A30", linestyle="--", alpha=0.7, label="Chosen K=4")
ax1.set_title("Elbow curve", fontweight="bold")
ax1.set_xlabel("Number of clusters (K)")
ax1.set_ylabel("Inertia")
ax1.legend(fontsize=9)

# --- Plot 2: Silhouette Score ---
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(K_range, sil_scores, "s-", color="#0F6E56", linewidth=2)
ax2.axvline(best_k, color="#D85A30", linestyle="--", alpha=0.7, label=f"Best K={best_k}")
ax2.set_title("Silhouette score", fontweight="bold")
ax2.set_xlabel("Number of clusters (K)")
ax2.set_ylabel("Score")
ax2.legend(fontsize=9)

# --- Plot 3: Segment Size ---
ax3 = fig.add_subplot(gs[0, 2])
seg_counts = df["segment"].value_counts()
bars = ax3.bar(range(len(seg_counts)), seg_counts.values,
               color=[COLORS[s] for s in seg_counts.index])
ax3.set_xticks(range(len(seg_counts)))
ax3.set_xticklabels([s.split()[0] for s in seg_counts.index], fontsize=9)
ax3.set_title("Segment size", fontweight="bold")
ax3.set_ylabel("Count")
for bar, val in zip(bars, seg_counts.values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
             str(val), ha="center", fontsize=9)

# --- Plot 4: Scatter — Frequency vs AOV ---
ax4 = fig.add_subplot(gs[1, :2])
for seg, grp in df.groupby("segment"):
    ax4.scatter(grp["purchase_frequency"], grp["avg_order_value"],
                color=COLORS[seg], alpha=0.6, s=40, label=seg, edgecolors="none")
ax4.set_title("Purchase frequency vs. avg order value", fontweight="bold")
ax4.set_xlabel("Purchase frequency (per month)")
ax4.set_ylabel("Avg order value ($)")
ax4.legend(fontsize=8, loc="upper right")

# --- Plot 5: Recency vs Frequency ---
ax5 = fig.add_subplot(gs[1, 2])
for seg, grp in df.groupby("segment"):
    ax5.scatter(grp["recency_days"], grp["purchase_frequency"],
                color=COLORS[seg], alpha=0.6, s=35, edgecolors="none")
ax5.set_title("Recency vs frequency", fontweight="bold")
ax5.set_xlabel("Days since last purchase")
ax5.set_ylabel("Purchase frequency")

# --- Plot 6: PCA 2D cluster view ---
ax6 = fig.add_subplot(gs[2, :2])
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
for seg, color in COLORS.items():
    mask = df["segment"] == seg
    ax6.scatter(X_pca[mask, 0], X_pca[mask, 1],
                color=color, alpha=0.6, s=35, label=seg, edgecolors="none")
ax6.set_title(f"PCA 2D cluster view  (variance explained: {sum(pca.explained_variance_ratio_)*100:.1f}%)",
              fontweight="bold")
ax6.set_xlabel("PCA component 1")
ax6.set_ylabel("PCA component 2")
ax6.legend(fontsize=8)

# --- Plot 7: Feature heatmap per segment ---
ax7 = fig.add_subplot(gs[2, 2])
heatmap_data = df.groupby("segment")[features].mean()
heatmap_norm = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min())
sns.heatmap(heatmap_norm, annot=heatmap_data.round(1), fmt=".1f",
            cmap="YlOrRd", ax=ax7, linewidths=0.5, annot_kws={"size": 8})
ax7.set_title("Segment feature heatmap\n(normalized)", fontweight="bold")
ax7.set_xticklabels([f.replace("_", "\n") for f in features], fontsize=8)
ax7.set_yticklabels([s.split()[0] for s in heatmap_norm.index], rotation=0, fontsize=8)

plt.savefig("customer_segments.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n✅ Chart 'customer_segments.png' -ஆக save ஆச்சு!")

# ─────────────────────────────────────────────
# STEP 7 — Result Export
# ─────────────────────────────────────────────
df.to_csv("segmented_customers.csv", index=False)
print("✅ 'segmented_customers.csv' save ஆச்சு!")
print("\nDone! 🎉")
