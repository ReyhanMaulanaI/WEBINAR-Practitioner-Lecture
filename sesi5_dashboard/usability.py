import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "usability_log.csv")

df = pd.read_csv(csv_path)
print("=== HASIL EVALUASI USABILITY TESTING ===")
print("\nSuccess rate per tugas (%):")
print((df.groupby("tugas")["berhasil"].mean() * 100).round(1))

print("\nRata-rata waktu pengerjaan (detik):")
print(df.groupby("tugas")["waktu_detik"].mean().round(1))

print("\nDetail ringkasan per tugas:")
ringkasan = df.groupby("tugas").agg(
    total_peserta=("peserta", "count"),
    tingkat_sukses_persen=("berhasil", lambda x: round(x.mean() * 100, 1)),
    rata_rata_waktu_detik=("waktu_detik", lambda x: round(x.mean(), 1))
)
print(ringkasan)
