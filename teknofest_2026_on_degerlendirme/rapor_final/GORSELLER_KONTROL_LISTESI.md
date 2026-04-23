# TomaTech TEKNOFEST 2026 Rapor Görselleri - Kontrol Listesi

## ✅ Hazır Olan Görseller (rapor_final/ klasöründe)

1. **hastallik_ornekleri_grid.png** - 10 hastalık sınıfının örnek görüntüleri (2x5 grid)
2. **model_karsilastirma.png** - Model doğruluk karşılaştırması bar grafiği
3. **sinif_dagilimi.png** - Veri seti sınıf dağılımı pasta grafiği
4. **sinif_performans.png** - Sınıf bazlı Precision/Recall/F1 bar grafiği
5. **egitim_sureci.png** - Validation accuracy eğitim süreci grafiği
6. **confusion_matrix_heatmap.png** - Confusion matrix heatmap
7. **confusion_matrix_original.png** - Orijinal confusion matrix (model çıktısı)
8. **per_class_metrics_original.png** - Orijinal sınıf metrikleri (model çıktısı)
9. **shuffle-block.png** - ShuffleNet V2 blok yapısı diyagramı
10. **top8_accuracy_bar.png** - En iyi 8 model karşılaştırması
11. **val_accuracy_curves.png** - Validation accuracy eğrileri

## ⚠️ Manuel Eklenmesi Gereken Görseller

### 1. Mobil Uygulama Ekran Görüntüleri (Mockup)
Rapora şu ekran tasarımları eklenmeli:

**Ekran 1: Ana Ekran**
- "Fotoğraf Çek" butonu (kamera simgesi ile)
- "Galeriden Seç" butonu
- "Geçmiş Kayıtlar" butonu
- Uygulama logosu ve adı (TomaTech)
- Yeşil/doğa temalı renk şeması

**Ekran 2: Kamera/Önizleme Ekranı**
- Kamera görünümü
- Çekim butonu
- Galeriye geçiş butonu
- Kılavuz çizgileri (yaprak ortalama yardımcısı)

**Ekran 3: Sonuç/Teşhis Ekranı**
- Çekilen/seçilen görüntü küçük resmi
- Tespit edilen hastalık adı (Türkçe, büyük font)
- Güven skoru yüzde olarak (örn: %98.5)
- Hastalık açıklaması (2-3 cümle)
- "Tedavi Önerileri" bölümü başlığı

**Ekran 4: Tedavi Önerileri Ekranı**
- Hastalık adı
- Maddeler halinde tedavi adımları:
  * İlaç/fungusit önerisi
  * Kültürel önlemler
  * Önleyici tedbirler
- "Yeni Teşhis" butonu
- "Kaydet" butonu

### 2. Sistem Mimarisi Diyagramı (Opsiyonel)
Basit bir akış diyagramı:
```
Kullanıcı → Fotoğraf → Ön İşleme → ShuffleNet V2 → Sonuç → Öneri
```

## 📁 Dosya Konumları

Tüm hazır görseller: 
```
/mnt/021630F41630E9F5/PROJECTS/torch/teknofest_2026_on_degerlendirme/rapor_final/
```

Hastalık örnek görüntüleri:
```
/mnt/021630F41630E9F5/PROJECTS/torch/teknofest_2026_on_degerlendirme/rapor_final/hastallik_ornekleri/
```

## 📝 Raporda Görsel Yerleştirme Notları

Rapordaki görsel yer tutucuları `[📷 Şekil X: ...]` veya `[📊 Şekil X: ...]` formatındadır.

DOCX formatına dönüştürürken:
1. Tüm görselleri uygun boyutta ekleyin
2. Şekil numaralarını kontrol edin
3. Şekil başlıklarını alt yazı olarak ekleyin
4. Görsellerin kalitesini kontrol edin (min 150 DPI)

---

Son Güncelleme: 30 Mart 2026
