import os
import warnings
import numpy as np
import geopandas as gpd
import folium
from shapely.geometry import box
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score

warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(42)

# ---------- 1. GRID 500 m ----------
minx, miny, maxx, maxy = 102.20, -3.90, 102.40, -3.70   # GANTI sesuai wilayah
step = 0.0045  # ~500 m
sel = [box(x, y, x + step, y + step)
       for x in np.arange(minx, maxx, step) for y in np.arange(miny, maxy, step)]
grid = gpd.GeoDataFrame(geometry=sel, crs="EPSG:4326")
grid["gx"] = grid.geometry.centroid.x
grid["gy"] = grid.geometry.centroid.y

# ---------- 2. FITUR (GANTI DENGAN DATA ASLI: raster DEMNAS, CHIRPS, OSM sungai) ----------
n = len(grid)
grid["elevasi"] = rng.gamma(2, 15, n)                   # meter
grid["lereng"] = rng.gamma(1.5, 3, n)                   # derajat
grid["jarak_sungai"] = rng.exponential(800, n)          # meter
grid["hujan"] = rng.normal(250, 60, n).clip(50)         # mm/bulan
z = (-0.04 * grid.elevasi - 0.15 * grid.lereng - 0.002 * grid.jarak_sungai
     + 0.01 * grid.hujan + rng.normal(0, 0.7, n))
grid["banjir_pernah"] = (z > np.percentile(z, 75)).astype(int)

fitur = ["elevasi", "lereng", "jarak_sungai", "hujan"]

# ---------- 3. SPATIAL BLOCK SPLIT (hindari kebocoran spasial) ----------
blok_x = (grid.gx > grid.gx.quantile(0.8))  # 20% blok timur jadi data uji
Xtr, ytr = grid.loc[~blok_x, fitur], grid.loc[~blok_x, "banjir_pernah"]
Xte, yte = grid.loc[blok_x, fitur], grid.loc[blok_x, "banjir_pernah"]

rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
rf.fit(Xtr, ytr)
p = rf.predict_proba(Xte)[:, 1]
print("F1 :", round(f1_score(yte, p > 0.5), 3))
print("AUC:", round(roc_auc_score(yte, p), 3))
print("Feature importances:", dict(zip(fitur, rf.feature_importances_.round(3))))

# ---------- 4. SKOR KERAWANAN ----------
grid["skor"] = rf.predict_proba(grid[fitur])[:, 1]
grid["kelas"] = np.select([grid.skor < .33, grid.skor < .66], ["Rendah", "Sedang"], default="Tinggi")
geojson_path = os.path.join(BASE_DIR, "kerawanan_banjir.geojson")
grid.to_file(geojson_path, driver="GeoJSON")

# ---------- 5. PETA WEB ----------
warna = {"Rendah": "#2ca25f", "Sedang": "#fec44f", "Tinggi": "#de2d26"}
m = folium.Map(location=[(miny + maxy) / 2, (minx + maxx) / 2], zoom_start=11)
folium.GeoJson(
    grid[["skor", "kelas", "geometry"]],
    style_function=lambda f: {"fillColor": warna.get(f["properties"]["kelas"], "#999999"),
                              "color": "none", "fillOpacity": 0.6},
    tooltip=folium.GeoJsonTooltip(fields=["kelas", "skor"]),
).add_to(m)
html_path = os.path.join(BASE_DIR, "sesi3_peta.html")
m.save(html_path)
print(f"Selesai: {geojson_path} & {html_path}")
