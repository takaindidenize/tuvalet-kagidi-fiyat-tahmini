# Tuvalet Kağıdı Fiyat Tahmini 🧻

Türkiye'deki tuvalet kağıdı ürünlerinin **rulo başına fiyatını** ürün özelliklerinden tahmin eden küçük bir makine öğrenmesi çalışması. Doğrusal Regresyon ve Random Forest modelleri karşılaştırılıyor, sonuçlar dört grafikle görselleştiriliyor.

Veriler Hepsiburada'daki 15 ürünün Mayıs 2026 fiyatlarından derlendi.

---

## Problem

Marketteki tuvalet kağıtlarında paket fiyatı tek başına bir şey anlatmıyor: 72'li bir paket 16'lı bir paketten pahalı olabilir ama rulo başına çok daha ucuza gelebilir. Bu projede asıl sorulan soru şu:

> Bir ürünün rulo sayısı, kat sayısı, markası, parfümlü olup olmaması ve premium segmentte olup olmaması biliniyorsa, rulo başına fiyatı ne kadar isabetli tahmin edilebilir?

Hedef değişken:

```
fiyat_per_rulo = paket_fiyati_tl / rulo_sayisi
```

---

## Veri Seti

15 ürün, 11 marka (Familia, Solo, Papia, Selpak, Silen, Komili, Maylo, Focus, Teno, Sofia, Only).

| Sütun | Açıklama |
|---|---|
| `urun_adi` | Ürünün tam adı |
| `marka` | Marka adı (modelde `LabelEncoder` ile sayısallaştırılıyor) |
| `rulo_sayisi` | Pakette kaç rulo var (16–72) |
| `kat_sayisi` | Kağıdın kat sayısı (1–4) |
| `paket_fiyati_tl` | Paketin raf fiyatı (TL) |
| `parfumlu` | Parfümlü mü (0/1) |
| `premium` | Premium segment mi (0/1) |
| `fiyat_per_rulo` | Hesaplanan hedef değişken (TL/rulo) |

`tuvalet_kagidi_dataset.csv` dosyası script çalıştırıldığında **üretilir** — veri setinin kendisi `tuvalet_kagidi_ml2.py` içinde tanımlıdır.

---

## Kurulum

```bash
git clone https://github.com/takaindidenize/tuvalet-kag-d--tahmini.git
cd tuvalet-kag-d--tahmini
pip install pandas numpy scikit-learn matplotlib
```

## Çalıştırma

```bash
python tuvalet_kagidi_ml2.py
```

---

## Script Ne Yapıyor?

1. **Veri hazırlığı** — 15 ürünlük tabloyu kurar, rulo başına fiyatı hesaplar, markaları `LabelEncoder` ile kodlar.
2. **Betimsel istatistik** — rulo başına fiyatın ortalaması, minimumu, maksimumu ve standart sapması; en ucuz ve en pahalı ürünü isimleriyle yazdırır.
3. **Model eğitimi** — veri %73 eğitim / %27 test olarak ayrılır (`random_state=42`), ardından iki model eğitilir:
   - `LinearRegression`
   - `RandomForestRegressor` (200 ağaç)
4. **Değerlendirme** — her iki model için MAE (Ortalama Mutlak Hata) ve R² skoru; ayrıca doğrusal regresyonun her özelliğe verdiği katsayılar.
5. **Örnek tahminler** — veri setinde bulunmayan 4 kurgusal ürün için iki modelin de tahmini yan yana basılır (ör. "Ekonomik 2 Katlı 32'li (parfümsüz)", "Lüks 4 Katlı 16'lı (parfümlü)").
6. **Görselleştirme** — 2x2'lik bir panel:
   - Marka bazlı ortalama rulo fiyatı (bar)
   - Rulo sayısı vs. fiyat, kat sayısına göre renklendirilmiş (scatter)
   - Gerçek vs. tahmin edilen fiyat, marka etiketleriyle (Random Forest)
   - Özellik önem sıralaması (feature importance)
7. **Kayıt** — işlenmiş veri seti `tuvalet_kagidi_dataset.csv` olarak (`utf-8-sig`) diske yazılır.

---

## Örnek Çıktı

```
HEPSIBURADA TUVALET KAĞIDI FİYAT VERİ SETİ
...

Temel İstatistikler (Rulo Başına Fiyat - TL):
Ortalama : ...
Min      : ...
Max      : ...

MODEL PERFORMANSI

 Doğrusal Regresyon:
MAE (Ortalama Mutlak Hata) : ... TL
R²  (Açıklama Gücü)        : ...

 Random Forest:
MAE (Ortalama Mutlak Hata) : ... TL
R²  (Açıklama Gücü)        : ...
```

---

## Sınırlar

Bu bir öğrenme projesi; sonuçları ciddi bir fiyat analizi olarak okumamak lazım:

- **15 satır çok az.** Test kümesi yalnızca 4 üründen oluşuyor, dolayısıyla R² ve MAE değerleri `random_state` değiştiğinde ciddi biçimde oynayabilir. Sağlam bir karşılaştırma için k-fold cross validation gerekir.
- **Fiyatlar tek zaman noktasından.** Mayıs 2026'daki raf fiyatları, üstelik tek bir platformdan. Kampanya dönemleri ve mağaza farkları veride yok.
- **Marka etiketi alfabetik kodlanıyor.** `LabelEncoder` markalara sırasız sayılar verdiği için doğrusal regresyonda marka katsayısı anlamlı yorumlanamaz; Random Forest bundan daha az etkileniyor. One-hot encoding daha doğru olur.
- **Gramaj yok.** Rulodaki yaprak sayısı veya metraj veride bulunmuyor; oysa fiyatı en çok açıklayan değişkenlerden biri muhtemelen bu.

## Yol Haritası

- [ ] Ürün sayısını artırmak, birden fazla siteden veri toplamak
- [ ] `marka` için one-hot encoding
- [ ] Cross-validation ile model karşılaştırması
- [ ] Rulo metrajı / yaprak sayısı özelliğini eklemek
- [ ] Grafikleri PNG olarak dışa aktarmak

## Kullanılan Kütüphaneler

pandas · numpy · scikit-learn · matplotlib
