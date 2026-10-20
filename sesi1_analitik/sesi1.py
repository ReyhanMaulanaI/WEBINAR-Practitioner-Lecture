import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RNG = np.random.default_rng(42)

# ---------- 0. DATA (GANTI DENGAN DATA ASLI: pd.read_csv("data.csv")) ----------
n = 500
df = pd.DataFrame({
    "id": [f"M{i:04d}" for i in range(n)],  # ID anonim
    "ipk_sem1": RNG.normal(3.0, 0.5, n).clip(0, 4),
    "ipk_sem2": RNG.normal(3.0, 0.5, n).clip(0, 4),
    "kehadiran_persen": RNG.normal(85, 10, n).clip(0, 100),
    "jam_kerja_paruh_waktu": RNG.integers(0, 25, n),
})
skor = 1.2 * df.ipk_sem1 + 1.0 * df.ipk_sem2 + 0.03 * df.kehadiran_persen - 0.04 * df.jam_kerja_paruh_waktu
df["lulus_tepat_waktu"] = (skor + RNG.normal(0, 0.6, n) > skor.median()).astype(int)
# sisipkan kotoran data agar realistis
df.loc[RNG.choice(n, 25, replace=False), "ipk_sem2"] = np.nan
df.loc[5, "kehadiran_persen"] = 400  # contoh salah input
df = pd.concat([df, df.iloc[:5]], ignore_index=True)  # duplikat

# ---------- 1. PEMBERSIHAN ----------
print("Duplikat:", df.duplicated().sum(), "| Missing:\n", df.isna().sum())
df = df.drop_duplicates()
df["ipk_sem2"] = df["ipk_sem2"].fillna(df["ipk_sem2"].median())

# tandai pencilan (IQR), JANGAN hapus otomatis -> tinjau manusia
num_cols = ["ipk_sem1", "ipk_sem2", "kehadiran_persen", "jam_kerja_paruh_waktu"]
flag = pd.Series(False, index=df.index)
for c in num_cols:
    q1, q3 = df[c].quantile([.25, .75]); iqr = q3 - q1
    flag |= (df[c] < q1 - 1.5 * iqr) | (df[c] > q3 + 1.5 * iqr)

outliers_path = os.path.join(BASE_DIR, "outliers_review.csv")
df[flag].to_csv(outliers_path, index=False)
print(f"{flag.sum()} baris pencilan disimpan ke {outliers_path} untuk ditinjau manusia")
df = df[df["kehadiran_persen"] <= 100].reset_index(drop=True)  # keputusan manusia: 400% = salah input

# ---------- 2. DESCRIPTIVE ----------
print("\n== DESCRIPTIVE ==")
print(df[num_cols + ["lulus_tepat_waktu"]].describe().round(2))
print("Persentase lulus tepat waktu:", round(df.lulus_tepat_waktu.mean() * 100, 1), "%")

# ---------- 3. DIAGNOSTIC ----------
print("\n== DIAGNOSTIC ==")
print(df[num_cols + ["lulus_tepat_waktu"]].corr()["lulus_tepat_waktu"].round(2))
print(df.groupby("lulus_tepat_waktu")[num_cols].mean().round(2))

# ---------- 4. PREDICTIVE ----------
X, y = df[num_cols], df["lulus_tepat_waktu"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
models = {"LogReg": LogisticRegression(max_iter=1000),
          "RandomForest": RandomForestClassifier(random_state=42)}
best, best_f1 = None, 0
for name, m in models.items():
    m.fit(Xtr, ytr)
    pred = m.predict(Xte)
    print(f"\n== {name} ==")
    print(classification_report(yte, pred, digits=3))
    print(confusion_matrix(yte, pred))
    f1 = classification_report(yte, pred, output_dict=True)["weighted avg"]["f1-score"]
    if f1 > best_f1: best, best_f1 = m, f1

# ---------- 5. PRESCRIPTIVE ----------
df["p_lulus"] = best.predict_proba(X)[:, 1]
def rekomendasi(p, kehadiran):
    if p < 0.35: return "Bimbingan intensif + monitoring bulanan"
    if p < 0.60: return "Bimbingan kelompok" + (" + tegur kehadiran" if kehadiran < 75 else "")
    return "Pantau rutin"
df["rekomendasi"] = [rekomendasi(p, k) for p, k in zip(df.p_lulus, df.kehadiran_persen)]
print("\n== PRESCRIPTIVE ==")
print(df.rekomendasi.value_counts())
rekomendasi_path = os.path.join(BASE_DIR, "hasil_rekomendasi.csv")
df[["id", "p_lulus", "rekomendasi"]].to_csv(rekomendasi_path, index=False)

plt.figure(figsize=(8, 5))
for val, group in df.groupby("lulus_tepat_waktu"):
    label = "Lulus Tepat Waktu" if val == 1 else "Terlambat"
    plt.hist(group["ipk_sem1"], bins=15, alpha=0.6, label=label)
plt.title("Distribusi IPK Semester 1 berdasarkan Kelulusan")
plt.xlabel("IPK Semester 1")
plt.ylabel("Frekuensi")
plt.legend()
plt.tight_layout()
hist_path = os.path.join(BASE_DIR, "ipk_hist.png")
plt.savefig(hist_path, dpi=150)
plt.close()
print(f"Grafik distribusi disimpan ke {hist_path}")
