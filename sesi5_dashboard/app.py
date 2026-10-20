import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Dasbor Risiko Banjir Bengkulu", layout="wide")
rng = np.random.default_rng(42)

# ---------- DATA (GANTI DENGAN DATA ASLI) ----------
kel = [(f"Kel-{i:02d}", rng.choice(["Selebar", "Gading Cempaka", "Kampung Melayu", "Muara Bangkahulu"]))
       for i in range(24)]
tanggal = pd.date_range("2026-09-01", periods=30)
rows = []
for nama, kec in kel:
    base = rng.uniform(5, 40)
    for t in tanggal:
        rows.append([nama, kec, t, max(0, rng.normal(base, 10))])
df = pd.DataFrame(rows, columns=["kelurahan", "kecamatan", "tanggal", "hujan_mm"])
rata = df.groupby(["kelurahan", "kecamatan"]).hujan_mm.mean().reset_index()
rata["risiko"] = pd.cut(rata.hujan_mm, [-1, 15, 30, 999], labels=["Rendah", "Sedang", "Tinggi"])
tindakan = {"Rendah": "Pantau berkala", "Sedang": "Siapkan kesiapsiagaan warga", "Tinggi": "Siaga evakuasi"}

# ---------- MODE TAMPILAN ----------
sederhana = st.sidebar.toggle("Tampilan sederhana (teks besar, kontras tinggi)")
if sederhana:
    st.markdown("<style>html,body,[class*='css']{font-size:22px !important;}</style>", unsafe_allow_html=True)

st.title("Dasbor Risiko Banjir Bengkulu")

# ---------- ZOOM & FILTER ----------
st.sidebar.header("Filter")
kec_pilih = st.sidebar.multiselect("Kecamatan", sorted(df.kecamatan.unique()), default=sorted(df.kecamatan.unique()))
rentang = st.sidebar.date_input("Rentang tanggal", (tanggal.min(), tanggal.max()))
risiko_pilih = st.sidebar.multiselect("Tingkat risiko", ["Rendah", "Sedang", "Tinggi"], default=["Rendah", "Sedang", "Tinggi"])

d = df[df.kecamatan.isin(kec_pilih)]
if len(rentang) == 2:
    d = d[(d.tanggal >= pd.Timestamp(rentang[0])) & (d.tanggal <= pd.Timestamp(rentang[1]))]
r = rata[rata.kecamatan.isin(kec_pilih) & rata.risiko.isin(risiko_pilih)]

# ---------- OVERVIEW ----------
c1, c2, c3 = st.columns(3)
c1.metric("Kelurahan risiko Tinggi", int((r.risiko == "Tinggi").sum()))
c2.metric("Rata-rata hujan (mm/hari)", f"{d.hujan_mm.mean():.1f}" if len(d) > 0 else "0.0")
c3.metric("Kelurahan dipantau", len(r))

warna = {"Rendah": "#1b9e77", "Sedang": "#e6ab02", "Tinggi": "#d95f02"}  # ramah buta warna
if len(r) > 0:
    fig = px.bar(r.sort_values("hujan_mm", ascending=False), x="kelurahan", y="hujan_mm",
                 color="risiko", color_discrete_map=warna, text="risiko",  # label teks, bukan hanya warna
                 labels={"hujan_mm": "Rata-rata hujan (mm/hari)"})
    fig.update_yaxes(rangemode="tozero")  # sumbu jujur
    fig.update_layout(template="simple_white")  # data-ink ratio tinggi
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Tidak ada data kelurahan yang cocok dengan filter yang dipilih.")

# ---------- DETAILS-ON-DEMAND ----------
st.subheader("Detail kelurahan")
if len(r) > 0:
    pilih = st.selectbox("Pilih kelurahan", r.kelurahan)
    if pilih:
        det = d[d.kelurahan == pilih]
        st.plotly_chart(px.line(det, x="tanggal", y="hujan_mm", title=f"Tren curah hujan {pilih}")
                        .update_layout(template="simple_white"), use_container_width=True)
        status = r[r.kelurahan == pilih].risiko.iloc[0]
        st.info(f"Status: **{status}**. Tindakan: {tindakan[status]}")
else:
    st.info("Pilih kelurahan akan muncul bila ada data yang difilter.")

st.download_button(
    label="Ekspor data (CSV)",
    data=r.to_csv(index=False),
    file_name="risiko_banjir.csv",
    mime="text/csv"
)
