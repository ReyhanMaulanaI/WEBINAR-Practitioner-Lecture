import time, os
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from sklearn.ensemble import RandomForestRegressor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def buat_data(seed, n=2000):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))  # suhu, kelembapan, angin, hotspot
    y = 30 + 8 * X[:, 3] - 5 * X[:, 2] + 2 * X[:, 0] + rng.normal(0, 3, n)  # PM2.5
    return X, y

def simulasi_skenario(seed):
    """Satu skenario = latih model + prediksi 7 hari (168 jam)."""
    X, y = buat_data(seed)
    model = RandomForestRegressor(n_estimators=40, random_state=seed, n_jobs=1)
    model.fit(X, y)
    X_future = np.random.default_rng(seed + 1).normal(size=(168, 4))
    return float(model.predict(X_future).mean())

def jalankan_serial(seeds):
    t = time.perf_counter()
    hasil = [simulasi_skenario(s) for s in seeds]
    return time.perf_counter() - t, hasil

def jalankan_paralel(seeds, workers):
    t = time.perf_counter()
    with Pool(workers) as p:
        hasil = p.map(simulasi_skenario, seeds)
    return time.perf_counter() - t, hasil

if __name__ == "__main__":
    seeds = list(range(32))
    print(f"Memulai simulasi {len(seeds)} skenario...")
    t_serial, ref = jalankan_serial(seeds)
    print(f"Serial: {t_serial:.2f} detik")

    maks = os.cpu_count() or 4
    daftar = [w for w in (1, 2, 4, 8, 16) if w <= maks]
    if len(daftar) == 1 and maks > 1:
        daftar.append(maks)
    speedups, efisiensi = [], []
    for w in daftar:
        t_par, hasil = jalankan_paralel(seeds, w)
        assert np.allclose(hasil, ref), "Hasil paralel harus sama dengan serial!"
        s = t_serial / t_par
        speedups.append(s)
        efisiensi.append(s / w)
        print(f"Worker={w:2d} | {t_par:6.2f}s | speedup={s:4.2f}x | efisiensi={s/w:4.2f}")

    plt.figure(figsize=(7, 5))
    plt.plot(daftar, speedups, "o-", label="Speedup aktual")
    plt.plot(daftar, daftar, "--", label="Ideal (Linear)")
    plt.xlabel("Jumlah worker (CPU cores)")
    plt.ylabel("Speedup")
    plt.title("Speedup Simulasi Kualitas Udara (PM2.5)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    speedup_path = os.path.join(BASE_DIR, "speedup.png")
    plt.savefig(speedup_path, dpi=150)
    plt.close()
    print(f"Grafik speedup disimpan ke {speedup_path}")

    # Hukum Amdahl: bagian serial membatasi speedup maksimum.
    # Speedup_maks = 1 / (f_serial + (1 - f_serial) / N)
