import time
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
torch.manual_seed(42)
rng = np.random.default_rng(42)
KELAS = ["pertanian_kecil", "jalan", "penebangan_selektif", "tambang"]
BAND, UKURAN = 6, 16   # S1: VV,VH | S2: B2,B3,B4,B8

# ---------- 1. DATA SINTETIS (GANTI DENGAN PATCH SENTINEL ASLI) ----------
def buat_patch(kelas):
    p = rng.normal(0.3, 0.05, (BAND, UKURAN, UKURAN))
    if kelas == 0:   # bercak kecil
        i, j = rng.integers(3, 12, 2)
        p[:, i:i+3, j:j+3] += 0.35
    elif kelas == 1: # garis memanjang
        r = rng.integers(2, 14)
        p[:, r, :] += 0.4
    elif kelas == 2: # titik-titik tersebar
        idx = rng.integers(0, UKURAN, (10, 2))
        p[:, idx[:, 0], idx[:, 1]] += 0.3
    else:            # area terang luas, radar tinggi
        p[:, 4:13, 4:13] += 0.5
        p[:2] += 0.2
    return p

def buat_data(n_per_kelas):
    X = np.stack([buat_patch(k) for k in range(4) for _ in range(n_per_kelas)]).astype("float32")
    y = np.repeat(np.arange(4), n_per_kelas).astype("int64")
    return X, y

X, y = buat_data(300)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ---------- 2. BENCHMARK RANDOM FOREST ----------
def fitur_stat(A):  # mean, std, min, max per band
    return np.concatenate([A.mean((2, 3)), A.std((2, 3)), A.min((2, 3)), A.max((2, 3))], axis=1)

def jalankan_rf(Xtr, ytr, Xte):
    t = time.perf_counter()
    rf = RandomForestClassifier(n_estimators=200, random_state=42).fit(fitur_stat(Xtr), ytr)
    return rf.predict_proba(fitur_stat(Xte)), time.perf_counter() - t

# ---------- 3. CNN KECIL ----------
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(BAND, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.k = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )

    def forward(self, x):
        return self.k(self.f(x))

def jalankan_cnn(Xtr, ytr, Xte, epoch=25):
    t = time.perf_counter()
    model = CNN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.long)

    for _ in range(epoch):
        for b in torch.randperm(len(xt)).split(64):
            opt.zero_grad()
            loss = lossf(model(xt[b]), yt[b])
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        prob = torch.softmax(model(torch.tensor(Xte, dtype=torch.float32)), 1).numpy()
    return prob, time.perf_counter() - t

# ---------- 4. EVALUASI + KELAS "TIDAK YAKIN" ----------
AMBANG = 0.6
def evaluasi(nama, prob, waktu):
    pred = prob.argmax(1)
    yakin = prob.max(1) >= AMBANG
    print(f"\n== {nama} == waktu latih {waktu:.2f}s")
    print(f"Accuracy (semua): {accuracy_score(yte, pred):.3f} | macro-F1: {f1_score(yte, pred, average='macro'):.3f}")
    acc_yakin = accuracy_score(yte[yakin], pred[yakin]) if yakin.sum() > 0 else 0.0
    print(f"Tidak yakin: {(~yakin).mean()*100:.1f}% | Accuracy pada yang yakin: {acc_yakin:.3f}")
    print("Confusion Matrix:")
    print(confusion_matrix(yte, pred))

p_rf, t_rf = jalankan_rf(Xtr, ytr, Xte)
evaluasi("Random Forest", p_rf, t_rf)

p_cnn, t_cnn = jalankan_cnn(Xtr, ytr, Xte)
evaluasi("CNN", p_cnn, t_cnn)

# ---------- 5. LEARNING CURVE ----------
print("\n== Learning curve (jumlah contoh berlabel vs macro-F1) ==")
for n in (20, 50, 100, 240):
    sub = np.concatenate([np.where(ytr == k)[0][:n] for k in range(4)])
    f_rf = f1_score(yte, jalankan_rf(Xtr[sub], ytr[sub], Xte)[0].argmax(1), average="macro")
    f_cnn = f1_score(yte, jalankan_cnn(Xtr[sub], ytr[sub], Xte)[0].argmax(1), average="macro")
    print(f"n/kelas={n:3d} | RF={f_rf:.3f} | CNN={f_cnn:.3f}")

# ---------- 6. DATA ASLI ----------
# Panduan alur data asli:
# 1. Google Earth Engine: ekspor patch S1 (VV,VH) + S2 (B2,B3,B4,B8) di sekitar titik peringatan -> GeoTIFF
# 2. Baca dengan rasterio: rasterio.open(path).read() -> array (6, H, W)
# 3. Label manual dari digitasi + validasi lapangan. Simpan tiap patch ke numpy (N, 6, 16, 16)
# 4. Ganti X, y di atas. Normalisasi per band (mean/std) sebelum masuk model.
