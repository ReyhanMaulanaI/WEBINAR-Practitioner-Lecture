import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.svm import SVR
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, r2_score, silhouette_score, precision_score, recall_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(42)

# ================= A. REGRESI HARGA TANAH =================
n = 800
land = pd.DataFrame({
    "luas_m2": rng.integers(80, 1000, n),
    "jarak_ke_pusat_km": rng.uniform(0.5, 20, n),
    "jarak_ke_jalan_m": rng.uniform(5, 800, n),
    "kecamatan": rng.choice(["Selebar", "Gading Cempaka", "Kampung Melayu", "Muara Bangkahulu"], n),
})
base = {"Selebar": 1.0, "Gading Cempaka": 1.2, "Kampung Melayu": 0.9, "Muara Bangkahulu": 1.1}
land["harga_juta"] = (land.luas_m2 * 2.5 * land.kecamatan.map(base)
                      - land.jarak_ke_pusat_km * 15 - land.jarak_ke_jalan_m * 0.1
                      + rng.normal(0, 60, n))   # GANTI DENGAN DATA ASLI

X, y = land.drop(columns="harga_juta"), land["harga_juta"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

prep = ColumnTransformer([
    ("num", StandardScaler(), ["luas_m2", "jarak_ke_pusat_km", "jarak_ke_jalan_m"]),
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["kecamatan"]),
])
models = {
    "Linear": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
    "SVR": SVR(C=100),
}
rows = []
for nama, mdl in models.items():
    pipe = make_pipeline(prep, mdl).fit(Xtr, ytr)
    r2_tr = r2_score(ytr, pipe.predict(Xtr)); r2_te = r2_score(yte, pipe.predict(Xte))
    cv = cross_val_score(make_pipeline(prep, mdl), X, y, cv=5, scoring="r2").mean()
    rows.append([nama, mean_absolute_error(yte, pipe.predict(Xte)), r2_te, r2_tr, cv])
    if r2_tr - r2_te > 0.15:
        print(f"PERINGATAN: {nama} berpotensi overfitting")
print("\n== A. EVALUASI MODEL REGRESI HARGA TANAH ==")
print(pd.DataFrame(rows, columns=["Model", "MAE", "R2_uji", "R2_latih", "R2_CV5"]).round(3))

# ================= B. KLASTERISASI =================
kel = pd.DataFrame({
    "kepadatan": np.r_[rng.normal(2000, 300, 40), rng.normal(6000, 800, 40), rng.normal(10000, 1000, 40)],
    "risiko": np.r_[rng.normal(0.2, 0.05, 40), rng.normal(0.5, 0.1, 40), rng.normal(0.8, 0.08, 40)],
    "pendapatan": np.r_[rng.normal(4, .5, 40), rng.normal(3, .5, 40), rng.normal(2, .4, 40)],
})
Z = StandardScaler().fit_transform(kel)
skor = {k: silhouette_score(Z, KMeans(k, n_init=10, random_state=42).fit_predict(Z)) for k in range(2, 7)}
k_best = max(skor, key=skor.get)
print("\n== B. KLASTERISASI KELURAHAN (K-MEANS) ==")
print("Silhouette scores:", {k: round(v, 3) for k, v in skor.items()}, "-> k terbaik =", k_best)
kel["klaster"] = KMeans(k_best, n_init=10, random_state=42).fit_predict(Z)
print(kel.groupby("klaster").mean().round(2))

P = PCA(2).fit_transform(Z)
plt.figure(figsize=(7, 5))
scatter = plt.scatter(P[:, 0], P[:, 1], c=kel.klaster, cmap="viridis", edgecolor="k", s=60)
plt.title("Visualisasi Klaster Kelurahan (PCA 2D)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(scatter, label="Klaster")
plt.tight_layout()
klaster_path = os.path.join(BASE_DIR, "klaster.png")
plt.savefig(klaster_path, dpi=150)
plt.close()
print(f"Grafik klaster disimpan ke {klaster_path}")

# ================= C. DETEKSI ANOMALI =================
print("\n== C. DETEKSI ANOMALI TRANSAKSI (ISOLATION FOREST) ==")
nt = 5000
trx = pd.DataFrame({"nominal": rng.lognormal(10, 0.6, nt), "jam": rng.integers(6, 23, nt),
                    "luar_negeri": rng.binomial(1, 0.01, nt)})
asli = np.zeros(nt, dtype=int)
idx = rng.choice(nt, 50, replace=False)
trx.loc[idx, "nominal"] *= 30; trx.loc[idx, "jam"] = 3; trx.loc[idx, "luar_negeri"] = 1; asli[idx] = 1

iso = IsolationForest(contamination=0.01, random_state=42).fit(trx)
pred = (iso.predict(trx) == -1).astype(int)
print("Precision:", round(precision_score(asli, pred), 3), "| Recall:", round(recall_score(asli, pred), 3))
