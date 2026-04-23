# TEKNOFEST 2026 
# HAVACILIK, UZAY VE TEKNOLOJİ FESTİVALİ
# TARIM TEKNOLOJİLERİ YARIŞMASI

---

# PROJE ÖN DEĞERLENDİRME RAPORU

---

## **TAKIM ADI:** TomaTech

## **PROJE ADI:** AKILLI TARIM İÇİN GERÇEK ZAMANLI MOBİL DOMATES HASTALIĞI TANI SİSTEMİ

## **BAŞVURU ID:** #4777326

## **TAKIM ID:** #877139

---

**Yarışma Kategorisi:** Üniversite ve Üzeri Seviyesi

**Proje Konusu:** Görüntü İşleme Sistemleri

**Başvuru Tarihi:** 18 Şubat 2026

---

## TAKIM ÜYELERİ

| Sıra | Ad Soyad | Görev |
|------|----------|-------|
| 1 | Demet ALICI KARACA | Takım Üyesi |
| 2 | Ferhat Kanar | Takım Üyesi |
| 3 | Seyit Ahmet Taş | Takım Üyesi |

---

## İÇİNDEKİLER

1. Proje Özeti / Proje Tanımı
2. Problemin Tanımı ve İhtiyaç Analizi
   - 2.1. Domates Yaprak Hastalıkları Sorunu
   - 2.2. Mevcut Tespit Yöntemlerinin Yetersizlikleri
   - 2.3. Hedef Kitle ve Kullanıcı İhtiyaçları
3. Çözüm Yaklaşımı
   - 3.1. Hastalık Tespiti Çözümü
   - 3.2. Mobil Uygulama Entegrasyonu
   - 3.3. Karar Destek Sistemi
4. Teknik Yöntem
   - 4.1. Veri Seti ve Ön İşleme
   - 4.2. Model Mimarisi: ShuffleNet V2
   - 4.3. Eğitim Stratejisi ve Hiperparametreler
   - 4.4. Değerlendirme Metrikleri
5. Deneysel Çalışmalar ve Sonuçlar
   - 5.1. Model Karşılaştırma Deneyleri
   - 5.2. En İyi Model Performansı
   - 5.3. Sınıf Bazlı Analiz
   - 5.4. Confusion Matrix Analizi
6. Mobil Uygulama Tasarımı
   - 6.1. Uygulama Mimarisi
   - 6.2. Kullanıcı Arayüzü Tasarımı
   - 6.3. Sistem Akış Diyagramı
7. Yenilikçi (İnovatif) Yönü
8. İş Planı ve Proje Takvimi
9. Riskler ve Önlemler
10. Beklenen Çıktılar ve Yaygın Etki
11. Sonuç ve Değerlendirme
12. Kaynakça
13. Ekler

---

# 1. PROJE ÖZETİ / PROJE TANIMI

Dünya çapında domates üretiminin en büyük tehditlerinden biri olan yaprak hastalıkları, verimi ciddi oranda azaltmakta ve ekonomik açıdan büyük kayıplara neden olmaktadır. Bu proje kapsamında, domates yaprak hastalıklarının erken ve doğru tespitini sağlayan, mobil cihazlarda çevrimdışı çalışabilen bir yapay zekâ destekli karar destek sistemi geliştirilmiştir.

**Temel Teknik Yaklaşım:**
- ShuffleNet V2 mimarisinin farklı ölçeklerinin (0.5x, 1.0x, 1.5x, 2.0x) sistematik karşılaştırması
- 10 sınıflı hastalık sınıflandırma (9 hastalık + sağlıklı)
- Transfer öğrenme ve veri artırma deneylerinin kapsamlı analizi

**En İyi Sonuç:** 
- **%99.72 test doğruluğu** (ShuffleNet V2 1.5x baseline model)
- **Weighted F1-Score: 0.9972**

Çözümün mobilde çevrimdışı çalışabilmesi hedeflenmiştir; bu sayede internet bağlantısı olmayan tarla koşullarında da kullanım mümkündür.

---

## 2. PROBLEM TANIMI VE İHTİYAÇ ANALİZİ

### 2.1 Problem Tanımı

Domates yaprak hastalıkları, dünya genelinde domates üretimini tehdit eden en önemli faktörlerden biridir. Bu hastalıklar:
- Verimi %20-50 oranında düşürebilir
- Ürün kalitesini ciddi şekilde etkileyebilir
- Ekonomik kayıplara ve gereksiz ilaç kullanımına yol açabilir

**Mevcut Tespit Yöntemlerinin Sorunları:**

| Sorun | Açıklama |
|-------|----------|
| **Uzman Bağımlılığı** | Geleneksel tespit süreci uzman gözlemine dayalıdır; zaman alıcı ve subjektiftir |
| **Geç Tespit** | Erken tespit gecikmesi, hastalığın yayılmasına ve ürün kaybına yol açar |
| **İnternet Bağımlılığı** | Bulut tabanlı çözümler internet bağlantısının zayıf olduğu sahalarda yetersiz kalır |
| **Karar Desteği Eksikliği** | Üreticiler, sahada anlık teşhis ve uygulanabilir müdahale öneri mekanizmasına ihtiyaç duyar |

### 2.2 Hedef Kitle

- Küçük ve orta ölçekli domates üreticileri
- Tarım danışmanları ve ziraat mühendisleri
- Kooperatif ve tarımsal örgütler
- Tarım teknolojileri araştırmacıları

### 2.3 Projenin Gerekliliği

Türkiye, dünyada domates üretiminde önemli bir yere sahiptir. FAO verilerine göre Türkiye, yıllık ~13 milyon ton domates üretimi ile dünya sıralamasında 4. sıradadır. Bu üretimin korunması için:

- Erken ve doğru hastalık tespiti kritik önem taşımaktadır
- Sahada hızlı karar destek sistemlerine ihtiyaç vardır
- Çevreye duyarlı, minimum ilaç kullanımını destekleyen çözümler gereklidir

---

## 3. PROJENİN AMACI VE HEDEFLERİ

### 3.1 Genel Amaç

Domates yaprak hastalıklarını mobil cihaz üzerinde hızlı, doğru ve çevrimdışı tespit eden bir yapay zekâ sistemi geliştirmek.

### 3.2 Spesifik Hedefler

| Hedef No | Hedef | Durum | Sonuç |
|----------|-------|-------|-------|
| H1 | 10 sınıfta yüksek tespit performansı (> %95 test doğruluğu) | ✅ Başarıldı | %99.72 |
| H2 | Parametre sayısı düşük, mobilde çalışabilir mimari | ✅ Başarıldı | ~5M parametre |
| H3 | Tespit sonucu ile birlikte hastalık bazlı öneri sunma | 🔄 Devam | Planlanan |
| H4 | Çiftçi odaklı, sade ve anlaşılır mobil arayüz | 🔄 Devam | Tasarım aşamasında |

---

## 4. TEKNİK YÖNTEM

### 4.1 Veri Seti

**Kaynak:** PlantVillage Domates Yaprak Görüntüleri Veri Seti

| Özellik | Değer |
|---------|-------|
| **Toplam Görüntü Sayısı** | 18,160 |
| **Sınıf Sayısı** | 10 |
| **Görüntü Formatı** | RGB |
| **Giriş Boyutu** | 224 × 224 piksel |
| **Eğitim/Test Oranı** | %80 / %20 |

**Sınıf Dağılımı:**

| Sınıf Adı | Türkçe Karşılık | Test Örnek Sayısı |
|-----------|-----------------|-------------------|
| Bacterial_spot | Bakteriyel Leke | 428 |
| Early_blight | Erken Yanıklık | 181 |
| Late_blight | Geç Yanıklık | 366 |
| Leaf_Mold | Yaprak Küfü | 191 |
| Septoria_leaf_spot | Septoria Yaprak Lekesi | 374 |
| Spider_mites | Kırmızı Örümcek | 335 |
| Target_Spot | Hedef Leke | 285 |
| Yellow_Leaf_Curl_Virus | Sarı Yaprak Kıvırcıklık Virüsü | 1047 |
| Mosaic_virus | Mozaik Virüsü | 87 |
| Healthy | Sağlıklı | 338 |

### 4.2 Ön İşleme

```
1. Görüntü yeniden boyutlandırma: 224 × 224 piksel
2. Piksel normalizasyonu: [0, 255] → [0, 1]
3. Standartlaştırma: [-1, 1] aralığına normalize etme
4. Veri artırma (augmentation deneylerinde):
   - Yatay çevirme (p=0.5)
   - Rastgele döndürme (±20°)
   - Parlaklık değişimi (0.8-1.2 faktör)
```

### 4.3 Model Mimarisi: ShuffleNet V2

ShuffleNet V2, mobil ve gömülü sistemler için optimize edilmiş hafif bir evrişimsel sinir ağı mimarisidir. Temel özellikleri:

**Mimari Avantajları:**
- **Channel Shuffle:** Kanal bilgisinin gruplar arasında karışmasını sağlar
- **Channel Split:** Hesaplama verimliliği için kanal ayrımı
- **Depthwise Convolution:** Parametre verimliliği
- **Global Average Pooling:** Tam bağlantılı katman yerine

**Mimari Detayları (1.5x Ölçek):**

| Aşama | Çıkış Kanalları | Tekrar |
|-------|-----------------|--------|
| Stem (Conv + MaxPool) | 24 | 1 |
| Stage 2 | 176 | 4 |
| Stage 3 | 352 | 8 |
| Stage 4 | 704 | 4 |
| Conv5 | 1024 | 1 |
| GAP + FC | 10 | 1 |

**Farklı Ölçeklerin Kanal Yapısı:**

| Ölçek | Stage 1 | Stage 2 | Stage 3 | Stage 4 | Conv5 |
|-------|---------|---------|---------|---------|-------|
| 0.5x | 24 | 48 | 96 | 192 | 1024 |
| 1.0x | 24 | 116 | 232 | 464 | 1024 |
| 1.5x | 24 | 176 | 352 | 704 | 1024 |
| 2.0x | 24 | 244 | 488 | 976 | 2048 |

### 4.4 Eğitim Parametreleri

| Parametre | Değer |
|-----------|-------|
| **Epoch** | 50 |
| **Batch Size** | 32 |
| **Optimizer** | SGD + Momentum (0.9) |
| **Başlangıç Learning Rate** | 0.01 |
| **LR Schedule** | Step Decay (her 20 epoch'ta ×0.1) |
| **Loss Function** | Sparse Categorical Cross-Entropy |
| **Framework** | TensorFlow 2.x / Keras |

### 4.5 Değerlendirme Metrikleri

- **Accuracy:** Genel doğruluk oranı
- **Precision (Weighted/Macro):** Pozitif tahmin doğruluğu
- **Recall (Weighted/Macro):** Gerçek pozitif yakalama oranı
- **F1-Score (Weighted/Macro):** Precision ve Recall'un harmonik ortalaması
- **Confusion Matrix:** Sınıf bazlı hata analizi

---

## 5. DENEYSEL SONUÇLAR

### 5.1 Model Karşılaştırma Tablosu

Aşağıdaki tablo, yapılan tüm deneylerin sonuçlarını özetlemektedir:

| Deney | Test Accuracy | Weighted F1 | Precision (W) | Recall (W) | Transfer Learning | Data Aug. |
|-------|---------------|-------------|---------------|------------|-------------------|-----------|
| **1.5x Baseline** | **0.9972** | **0.9972** | 0.9972 | 0.9972 | Kapalı | Kapalı |
| 2.0x Baseline | 0.9961 | 0.9961 | 0.9961 | 0.9961 | Kapalı | Kapalı |
| 1.0x Baseline | 0.9956 | 0.9956 | 0.9956 | 0.9956 | Kapalı | Kapalı |
| 2.0x Aug | 0.9920 | 0.9920 | 0.9920 | 0.9920 | Kapalı | Açık |
| 1.5x Aug | 0.9909 | 0.9909 | 0.9909 | 0.9909 | Kapalı | Açık |
| 1.0x Aug | 0.9851 | 0.9851 | 0.9851 | 0.9851 | Kapalı | Açık |
| 0.5x Aug | 0.9829 | 0.9829 | 0.9830 | 0.9829 | Kapalı | Açık |
| 0.5x Baseline | 0.9749 | 0.9749 | 0.9750 | 0.9749 | Kapalı | Kapalı |

### 5.2 En İyi Model Detaylı Sonuçları (ShuffleNet V2 1.5x Baseline)

**Genel Metrikler:**

| Metrik | Değer |
|--------|-------|
| **Test Accuracy** | 0.9972 (%99.72) |
| **Weighted Precision** | 0.9972 |
| **Weighted Recall** | 0.9972 |
| **Weighted F1-Score** | 0.9972 |
| **Macro Precision** | 0.9966 |
| **Macro Recall** | 0.9956 |
| **Macro F1-Score** | 0.9961 |
| **Toplam Test Örneği** | 3,632 |

**Sınıf Bazlı Performans:**

| Sınıf | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Bacterial_spot | 0.9953 | 1.0000 | 0.9977 | 428 |
| Early_blight | 0.9833 | 0.9779 | 0.9806 | 181 |
| Late_blight | 0.9973 | 0.9945 | 0.9959 | 366 |
| Leaf_Mold | 1.0000 | 0.9895 | 0.9947 | 191 |
| Septoria_leaf_spot | 0.9973 | 0.9973 | 0.9973 | 374 |
| Spider_mites | 0.9970 | 0.9970 | 0.9970 | 335 |
| Target_Spot | 0.9965 | 1.0000 | 0.9982 | 285 |
| Yellow_Leaf_Curl | 0.9990 | 1.0000 | 0.9995 | 1047 |
| Mosaic_virus | 1.0000 | 1.0000 | 1.0000 | 87 |
| Healthy | 1.0000 | 1.0000 | 1.0000 | 338 |

### 5.3 Confusion Matrix

**[📊 Grafik 1: Confusion Matrix - ShuffleNet V2 1.5x Baseline]**
*Dosya Konumu: shufflenet-v2-tensorflow/checkpoints_tomato_1_5x_baseline/confusion_matrix.png*

```
Confusion Matrix (1.5x Baseline - En İyi Model):
                    Tahmin
                    BS   EB   LB   LM   SS   SM   TS  YLC  MV   H
Gerçek:
Bacterial_spot(BS) 428   0    0    0    0    0    0    0    0   0
Early_blight(EB)     1 177    1    0    1    0    0    1    0   0
Late_blight(LB)      0   2  364    0    0    0    0    0    0   0
Leaf_Mold(LM)        1   0    0  189    0    1    0    0    0   0
Septoria(SS)         0   1    0    0  373    0    0    0    0   0
Spider_mites(SM)     0   0    0    0    0  334    1    0    0   0
Target_Spot(TS)      0   0    0    0    0    0  285    0    0   0
Yellow_Leaf(YLC)     0   0    0    0    0    0    0 1047    0   0
Mosaic_virus(MV)     0   0    0    0    0    0    0    0   87   0
Healthy(H)           0   0    0    0    0    0    0    0    0 338
```

### 5.4 Eğitim Sürecinin Analizi

**[📊 Grafik 2: Validation Accuracy Eğrileri]**
*Dosya Konumu: teknofest_2026_on_degerlendirme/assets/val_accuracy_curves.png*

**Eğitim İstatistikleri (1.5x Baseline):**

| Epoch Aralığı | Learning Rate | Val Accuracy Aralığı |
|---------------|---------------|----------------------|
| 0-19 | 0.01 | 0.20 → 0.93 |
| 20-39 | 0.001 | 0.98 → 0.99 |
| 40-49 | 0.0001 | 0.99 → 0.99 |

**En İyi Validation Accuracy:** 0.9810 (Epoch 17)

### 5.5 Model Karşılaştırma Grafikleri

**[📊 Grafik 3: Top 8 Deney Test Doğruluğu Bar Grafiği]**
*Dosya Konumu: teknofest_2026_on_degerlendirme/assets/top8_accuracy_bar.png*

**[📊 Grafik 4: Sınıf Bazlı Metrikler (Per-Class Metrics)]**
*Dosya Konumu: checkpoints_tomato_1_5x_aug/per_class_metrics.png*

### 5.6 Önemli Gözlemler

1. **1.5x Baseline En İyi Performans:** Sıfırdan eğitilen 1.5x ölçek model, en yüksek test performansını (%99.72) vermiştir.

2. **Data Augmentation Etkisi:** Veri artırma uygulanan modeller, baseline modellere göre biraz daha düşük performans göstermiştir. Bu durum, veri setinin yeterli çeşitliliğe sahip olduğunu ve ek augmentation'ın overfitting'e yol açtığını göstermektedir.

3. **Transfer Learning Sorunu:** ImageNet ön eğitimli ağırlıklarla transfer öğrenme senaryolarında, sınıf collapse problemi gözlemlenmiştir. Bu nedenle sıfırdan eğitim yaklaşımı tercih edilmiştir.

4. **Ölçek Karşılaştırması:** 1.5x ve 2.0x ölçekler çok yakın performans gösterirken, 0.5x en hafif ancak en düşük performanslı modeldir.

---

## 6. YENİLİKÇİ YÖNLER

### 6.1 Teknik Yenilikler

| Yenilik | Açıklama |
|---------|----------|
| **Hafif CNN Yaklaşımı** | ShuffleNet V2 ile mobil uygunluk ön planda tutulmuştur |
| **Çoklu Ölçek Deneyi** | 0.5x-2.0x arası sistematik karşılaştırma yapılmıştır |
| **Çevrimdışı Çalışma** | İnternet gerektirmeden sahada kullanım hedeflenmiştir |
| **Karar Destek Katmanı** | Teşhis sonucuna ek olarak tedavi/yönetim önerisi sunulması planlanmıştır |

### 6.2 Uygulama Yenilikleri

- **On-Device Inference:** Model, mobil cihazda yerel olarak çalışacak şekilde optimize edilecektir
- **Gerçek Zamanlı Tespit:** Kamera akışından anlık hastalık tespiti
- **Pratik Öneri Sistemi:** Hastalık bazlı müdahale önerileri

### 6.3 Mevcut Çözümlerden Farklar

| Özellik | Mevcut Çözümler | Bizim Projemiz |
|---------|-----------------|----------------|
| İnternet Gereksinimi | Genellikle bulut tabanlı | Çevrimdışı çalışır |
| Model Boyutu | Büyük modeller | Hafif (ShuffleNet V2) |
| Doğruluk | %85-95 | %99.72 |
| Öneri Sistemi | Genellikle yok | Entegre |
| Hedef Platform | Web/Desktop | Mobil odaklı |

---

## 7. MOBİL UYGULAMA ENTEGRASYOnu

### 7.1 Planlanan Mobil Akış

```
┌─────────────────┐
│ Kullanıcı Girişi│
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Görüntü Alımı                  │
│  - Kamera ile çekim             │
│  - Galeriden seçim              │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Ön İşleme                      │
│  - 224×224 boyutlandırma        │
│  - Normalizasyon                │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  On-Device Model Çağrısı        │
│  - TensorFlow Lite inference    │
│  - Sınıf tahmini                │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Sonuç Ekranı                   │
│  - Hastalık adı                 │
│  - Güven skoru (%)              │
│  - Pratik müdahale önerileri    │
└─────────────────────────────────┘
```

### 7.2 Mobil Ekran Tasarımları

**[📱 Ekran 1: Ana Ekran]**
*Açıklama: Kamera aç / galeriden seç butonları, sade kullanım akışı*
> **Not:** Temsili mobil ekran görüntüsü eklenecek

**[📱 Ekran 2: Tanı Sonucu Ekranı]**
*Açıklama: Tahmin edilen hastalık adı, güven yüzdesi, en yakın 2 alternatif sınıf*
> **Not:** Temsili mobil ekran görüntüsü eklenecek

**[📱 Ekran 3: Öneri Ekranı]**
*Açıklama: Hastalığa özel müdahale adımları (kısa madde listesi)*
> **Not:** Temsili mobil ekran görüntüsü eklenecek

**[📱 Ekran 4: Geçmiş Kayıt Ekranı]**
*Açıklama: Tarih, tahmin, konum veya not bilgisi*
> **Not:** Temsili mobil ekran görüntüsü eklenecek

### 7.3 Teknik Gereksinimler

| Platform | Gereksinim |
|----------|------------|
| Android | API Level 21+ (Android 5.0+) |
| iOS | iOS 12.0+ |
| RAM | Minimum 2GB |
| Depolama | ~50MB (uygulama + model) |

---

## 8. İŞ PLANI VE TAKVİM

### 8.1 İş Paketleri

| WP | Başlık | Açıklama | Durum |
|----|--------|----------|-------|
| WP1 | Veri Düzenleme | Veri seti hazırlığı ve etiket kalite kontrolü | ✅ Tamamlandı |
| WP2 | Model Eğitimi | Hiperparametre optimizasyonu ve model karşılaştırması | ✅ Tamamlandı |
| WP3 | Mobil Entegrasyon | TFLite dönüşümü ve mobil uygulama geliştirme | 🔄 Devam Ediyor |
| WP4 | Saha Testi | Gerçek koşullarda doğrulama ve raporlama | 📋 Planlanan |

### 8.2 Takvim

| Dönem | Aktivite |
|-------|----------|
| Şubat 2026 | Proje planlaması, veri seti hazırlığı |
| Ocak-Şubat 2026 | Model eğitimi ve optimizasyon deneyleri |
| Mart 2026 | PDR raporu hazırlığı, mobil tasarım |
| Nisan-Mayıs 2026 | Mobil uygulama geliştirme |
| Haziran 2026 | Saha testleri ve iyileştirmeler |
| Temmuz-Ağustos 2026 | Yarı final sunumu hazırlıkları |

---

## 9. RİSKLER VE ÖNLEMLER

| Risk | Olasılık | Etki | Önlem |
|------|----------|------|-------|
| Sınıflar arası dengesizlik | Orta | Yüksek | Sınıf bazlı metrik takibi, weighted loss |
| Aşırı öğrenme (overfitting) | Düşük | Orta | Early stopping, dropout, regularization |
| Mobilde gecikme | Orta | Orta | Model quantization, ölçek optimizasyonu |
| Transfer öğrenme başarısızlığı | Yüksek | Orta | Sıfırdan eğitim alternatifi (uygulandı) |
| Farklı çevre koşullarında performans düşüşü | Orta | Yüksek | Saha veri toplama, fine-tuning |

---

## 10. YAYGIN ETKİ VE KATMA DEĞER

### 10.1 Tarımsal Faydalar

- **Erken Teşhis:** Hastalığın erken aşamada tespiti ile yayılmanın önlenmesi
- **Verim Artışı:** Zamanında müdahale ile ürün kaybının azaltılması
- **Maliyet Düşürme:** Gereksiz ilaç kullanımının önlenmesi

### 10.2 Çevresel Faydalar

- **Azaltılmış Pestisit Kullanımı:** Hedefli tedavi ile kimyasal kullanımının minimizasyonu
- **Sürdürülebilir Tarım:** Çevre dostu üretim pratiklerinin desteklenmesi

### 10.3 Sosyal Faydalar

- **Erişilebilirlik:** Küçük çiftçilerin de teknolojiye ulaşabilmesi
- **Karar Destek:** Uzman olmayan kullanıcılar için kolay karar mekanizması
- **Dijital Okuryazarlık:** Tarım sektöründe teknoloji kullanımının yaygınlaştırılması

### 10.4 Ekonomik Etki

| Etki Alanı | Beklenen Katkı |
|------------|----------------|
| Verim Kaybı Azaltma | %10-20 |
| İlaç Maliyeti Düşüşü | %15-25 |
| İşçilik Tasarrufu | %30-40 |

---

## 11. SONUÇ

Bu proje, domates yaprak hastalıklarının mobil cihazlarda gerçek zamanlı ve çevrimdışı tespitine odaklanan, yüksek doğruluklu bir görüntü işleme çözümüdür.

**Öne Çıkan Başarılar:**

✅ ShuffleNet V2 tabanlı deneylerde **%99.72** seviyesine ulaşan test doğruluğu

✅ 4 farklı model ölçeğinin (0.5x, 1.0x, 1.5x, 2.0x) sistematik karşılaştırması

✅ 10 sınıflı hastalık sınıflandırması için kapsamlı metrik analizi

✅ Mobilde çevrimdışı kullanıma uygun hafif mimari seçimi

**Sonraki Adımlar:**

1. TensorFlow Lite dönüşümü ve mobil optimizasyon
2. Android/iOS uygulama geliştirme
3. Hastalık bazlı öneri veritabanı oluşturma
4. Saha testleri ve kullanıcı geri bildirimi toplama

Bu yapı ile hem teknik başarı hem de tarımsal uygulanabilirlik birlikte sağlanmaktadır. Projenin, Türk tarım sektöründe dijitalleşme sürecine önemli bir katkı sunması hedeflenmektedir.

---

## 12. KAYNAKLAR

1. PlantVillage Dataset - Penn State University
2. ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design (Ma et al., 2018)
3. TensorFlow Documentation - https://www.tensorflow.org/
4. TEKNOFEST Tarım Teknolojileri Yarışması Şartnamesi 2026

---

## 13. EKLER

### Ek 1: Grafik ve Şekil Listesi

| No | Grafik/Şekil | Konum |
|----|--------------|-------|
| 1 | Confusion Matrix (1.5x Baseline) | shufflenet-v2-tensorflow/checkpoints_tomato_1_5x_baseline/confusion_matrix.png |
| 2 | Per-Class Metrics (1.5x Baseline) | shufflenet-v2-tensorflow/checkpoints_tomato_1_5x_baseline/per_class_metrics.png |
| 3 | Validation Accuracy Curves | teknofest_2026_on_degerlendirme/assets/val_accuracy_curves.png |
| 4 | Top 8 Accuracy Bar Chart | teknofest_2026_on_degerlendirme/assets/top8_accuracy_bar.png |
| 5 | Confusion Matrix (1.5x Aug) | checkpoints_tomato_1_5x_aug/confusion_matrix.png |
| 6 | Per-Class Metrics (1.5x Aug) | checkpoints_tomato_1_5x_aug/per_class_metrics.png |
| 7 | Confusion Matrix (2.0x Aug) | checkpoints_tomato_2_0x_aug/confusion_matrix.png |
| 8 | Per-Class Metrics (2.0x Aug) | checkpoints_tomato_2_0x_aug/per_class_metrics.png |
| 9 | Model Comparison Chart | shufflenet-v2-tensorflow/model_comparison.png |

### Ek 2: Model Dosyaları

| Model | Dosya Konumu | Boyut |
|-------|--------------|-------|
| 1.5x Baseline (En İyi) | shufflenet-v2-tensorflow/checkpoints_tomato_1_5x_baseline/best_model.keras | ~20MB |
| 1.5x Aug | checkpoints_tomato_1_5x_aug/best_model.keras | ~20MB |
| 2.0x Aug | checkpoints_tomato_2_0x_aug/best_model.keras | ~42MB |

### Ek 3: Eğitim Logları

Tüm deneylerin eğitim logları CSV formatında ilgili checkpoint klasörlerinde mevcuttur:
- `training_log.csv`: Epoch bazlı accuracy, loss, learning rate
- `test_results.csv`: Test metrikleri ve confusion matrix

---

**Rapor Hazırlama Tarihi:** 30 Mart 2026

**Takım:** TomaTech

**İletişim:** [Takım e-posta adresi eklenecek]

---

*Bu rapor, TEKNOFEST 2026 Tarım Teknolojileri Yarışması Proje Ön Değerlendirme (PDR) aşaması için hazırlanmıştır.*
