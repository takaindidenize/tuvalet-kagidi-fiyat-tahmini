import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

data = {
    "urun_adi": [
        "Familia Plus Natural 40'lı",
        "Solo Bambu Katkılı 40'lı",
        "Papia Pure & Soft 4 Katlı 32'li",
        "Selpak Comfort 3 Katlı 32'li",
        "Silen Ekonomik 2 Katlı 32'li",
        "Komili Natural 3 Katlı 24'lü",
        "Maylo Premium 3 Katlı 40'lı",
        "Focus Ekonomik 2 Katlı 48'li",
        "Solo Bambu 3 Katlı 24'lü",
        "Familia Plus 2 Katlı 32'li",
        "Selpak Soft 3 Katlı 16'lı",
        "Papia Soft 2 Katlı 48'li",
        "Teno Ekonomik 1 Katlı 72'li",
        "Sofia Premium 4 Katlı 16'lı",
        "Only Ekonomi 2 Katlı 64'lü",
    ],
    "marka": [
        "Familia", "Solo", "Papia", "Selpak", "Silen",
        "Komili", "Maylo", "Focus", "Solo", "Familia",
        "Selpak", "Papia", "Teno", "Sofia", "Only",
    ],
    "rulo_sayisi": [40, 40, 32, 32, 32, 24, 40, 48, 24, 32, 16, 48, 72, 16, 64],
    "kat_sayisi":  [3, 3, 4, 3, 2, 3, 3, 2, 3, 2, 3, 2, 1, 4, 2],
    "paket_fiyati_tl": [
        304.42, 329.90, 389.00, 345.00, 189.90,
        249.00, 375.00, 299.00, 199.00, 265.00,
        179.00, 320.00, 310.00, 199.00, 390.00,
    ],
    "parfumlu": [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0],
    "premium":  [1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0],
}

df = pd.DataFrame(data)
df["fiyat_per_rulo"] = (df["paket_fiyati_tl"] / df["rulo_sayisi"]).round(2)

le = LabelEncoder()
df["marka_enc"] = le.fit_transform(df["marka"])

print("\nHEPSIBURADA TUVALETKAĞIDIFİYAT VERİ SETİ")
print(df[["urun_adi", "marka", "rulo_sayisi", "kat_sayisi",
          "paket_fiyati_tl", "fiyat_per_rulo"]].to_string(index=False))

print(f"\nTemel İstatistikler (Rulo Başına Fiyat - TL):")
print(f"Ortalama : {df['fiyat_per_rulo'].mean():.2f} TL")
print(f"Min      : {df['fiyat_per_rulo'].min():.2f} TL  ({df.loc[df['fiyat_per_rulo'].idxmin(),'urun_adi']})")
print(f"Max      : {df['fiyat_per_rulo'].max():.2f} TL  ({df.loc[df['fiyat_per_rulo'].idxmax(),'urun_adi']})")
print(f"Std      : {df['fiyat_per_rulo'].std():.2f} TL")

features = ["rulo_sayisi", "kat_sayisi", "parfumlu", "premium", "marka_enc"]
X = df[features]
y = df["fiyat_per_rulo"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.27, random_state=42
)

#Doğrusal Regresyon
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

#Random Forest
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("   MODEL PERFORMANSI")
for name, pred in [("Doğrusal Regresyon", y_pred_lr), ("Random Forest", y_pred_rf)]:
    mae = mean_absolute_error(y_test, pred)
    r2  = r2_score(y_test, pred)
    print(f"\n  {name}:")
    print(f"MAE (Ortalama Mutlak Hata) : {mae:.2f} TL")
    print(f"R²  (Açıklama Gücü)        : {r2:.2f}")

#Katsayılar
print("\nDoğrusal Regresyon Katsayıları:")
for feat, coef in zip(features, lr.coef_):
    print(f"   {feat:<15} → {coef:+.4f}")

#ÖRNEK TAHMİN
print("\n   TAHMİN ÖRNEKLERİ (Rulo Başına Fiyat)")
ornek_urunler = pd.DataFrame({
    "urun_tanimi": [
        "Ekonomik 2 Katlı 32'li (parfümsüz)",
        "Premium 3 Katlı 40'lı (parfümlü)",
        "Toplu 1 Katlı 72'li (parfümsüz)",
        "Lüks 4 Katlı 16'lı (parfümlü)",
    ],
    "rulo_sayisi": [32, 40, 72, 16],
    "kat_sayisi":  [2,  3,  1,  4],
    "parfumlu":    [0,  1,  0,  1],
    "premium":     [0,  1,  0,  1],
    "marka_enc":   [4,  6,  8,  9],
})

tahmin_lr = lr.predict(ornek_urunler[features])
tahmin_rf = rf.predict(ornek_urunler[features])

for i, row in ornek_urunler.iterrows():
    print(f"\n{row['urun_tanimi']}")
    print(f"Doğrusal Regresyon  → {tahmin_lr[i]:.2f} TL/rulo")
    print(f"Random Forest       → {tahmin_rf[i]:.2f} TL/rulo")

#GRAFİKLER
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Hepsiburada – Tuvalet Kağıdı Fiyat Analizi (Mayıs 2026)",
             fontsize=14, fontweight='bold', y=1.01)

#Marka bazlı fiyat
ax1 = axes[0, 0]
marka_ort = df.groupby("marka")["fiyat_per_rulo"].mean().sort_values(ascending=False)
bars = ax1.bar(marka_ort.index, marka_ort.values,
               color=plt.cm.Set2(np.linspace(0, 1, len(marka_ort))))
ax1.set_title("Marka Bazlı Ort. Rulo Fiyatı (TL)", fontweight='bold')
ax1.set_xlabel("Marka"); ax1.set_ylabel("TL / Rulo")
ax1.tick_params(axis='x', rotation=45)
for bar, val in zip(bars, marka_ort.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f"{val:.2f}", ha='center', va='bottom', fontsize=8)

#Kat sayısı vs rulo fiyatı
ax2 = axes[0, 1]
for kat in sorted(df["kat_sayisi"].unique()):
    grp = df[df["kat_sayisi"] == kat]
    ax2.scatter(grp["rulo_sayisi"], grp["fiyat_per_rulo"],
                label=f"{kat} Katlı", s=100, alpha=0.8)
ax2.set_title("Rulo Sayısı vs Fiyat (Kat Sayısına Göre)", fontweight='bold')
ax2.set_xlabel("Rulo Sayısı"); ax2.set_ylabel("TL / Rulo")
ax2.legend(title="Kat Sayısı"); ax2.grid(True, alpha=0.3)

#Tahminler
ax3 = axes[1, 0]
y_pred_all_rf = rf.predict(X)
ax3.scatter(y, y_pred_all_rf, color='steelblue', alpha=0.8, s=80, edgecolors='white')
mn, mx = y.min()-0.5, y.max()+0.5
ax3.plot([mn, mx], [mn, mx], 'r--', lw=2, label='Mükemmel Tahmin')
for i, row in df.iterrows():
    ax3.annotate(row["marka"], (y[i], y_pred_all_rf[i]),
                 textcoords="offset points", xytext=(4, 4), fontsize=7, alpha=0.7)
ax3.set_title("Gerçek vs Tahmin (Random Forest)", fontweight='bold')
ax3.set_xlabel("Gerçek Fiyat (TL/Rulo)"); ax3.set_ylabel("Tahmin (TL/Rulo)")
ax3.legend(); ax3.grid(True, alpha=0.3)

#feature importance
ax4 = axes[1, 1]
feat_imp = pd.Series(rf.feature_importances_,
                     index=["Rulo Sayısı", "Kat Sayısı", "Parfümlü", "Premium", "Marka"])
feat_imp = feat_imp.sort_values()
colors = ['#e74c3c' if v == feat_imp.max() else '#3498db' for v in feat_imp.values]
feat_imp.plot(kind='barh', ax=ax4, color=colors)
ax4.set_title("Özellik Önem Sıralaması (Random Forest)", fontweight='bold')
ax4.set_xlabel("Önem Skoru")
ax4.grid(True, alpha=0.3, axis='x')


#csv kaydetme
df_cikti = df.drop(columns=["marka_enc"])
df_cikti.to_csv("tuvalet_kagidi_dataset.csv",
                index=False, encoding="utf-8-sig")
print("Dataset kaydedildi → tuvalet_kagidi_dataset.csv")
print("\nTamamlandı! ")
