# 2026 Tarım Teknolojileri Yarışması (Universite ve Uzeri)
## Proje On Degerlendirme Formu - Taslak (TomaTech)

## 1) Takim ve Basvuru Bilgileri
- Takim Adi: TomaTech
- Takim ID: #877139
- Basvuru ID: #4777326
- Yarisma: 2026 Tarim Teknolojileri Yarismasi Universite ve Uzeri Seviyesi
- Proje Basligi: Akilli Tarim icin Gercek Zamanli Mobil Domates Hastaligi Tani Sistemi
- Proje Konusu: Goruntu Isleme Sistemleri
- Takim Uyeleri: Demet ALICI KARACA, ferhat Kanar, Seyit Ahmet Tas

## 2) Proje Ozeti
Domates uretiminde verim kaybina neden olan yaprak hastaliklari, erken ve dogru tespit edilmediginde hem ekonomik kayip hem de gereksiz ilac kullanimi olusturmaktadir. Bu proje ile, domates yaprak goruntulerinden hastaligi gercek zamanli tespit eden, hafif ve mobil cihazlarda calisabilen bir yapay zeka tabanli karar destek sistemi gelistirilmistir.

Temel teknik yaklasim, ShuffleNet V2 mimarisinin farkli olceklerinin (0.5x, 1.0x, 1.5x, 2.0x) karsilastirilmasi ve en iyi dogruluk-hiz dengesinin secilmesi uzerinedir. Test sonuclarinda en iyi modelde %99.72 dogruluk (1.5x baseline) elde edilmistir. Cozumun mobilde cevrimdisi calisabilmesi hedeflenmistir; bu sayede internet baglantisi olmayan tarla kosullarinda da kullanim mumkundur.

## 3) Problem Tanimi ve Ihtiyac Analizi
- Geleneksel tespit sureci uzman gozlemine baglidir; zaman alici ve subjektiftir.
- Erken tespit gecikmesi, hastaligin yayilmasina ve urun kaybina yol acar.
- Internet baglantisinin zayif oldugu sahalarda bulut bagimli cozumler yetersiz kalir.
- Ureticiler, sahada anlik teshis ve uygulanabilir mudahale oneri mekanizmasina ihtiyac duyar.

## 4) Projenin Amaci ve Hedefleri
- Amac: Domates yaprak hastaliklarini mobil cihaz uzerinde hizli, dogru ve cevrimdisi tespit eden bir sistem gelistirmek.
- Hedef 1: 10 sinifta yuksek tespit performansi elde etmek (hedef > %95 test dogrulugu).
- Hedef 2: Parametre sayisi dusuk bir mimari ile mobilde calisabilirlik saglamak.
- Hedef 3: Tespit sonucu ile birlikte kullaniciya hastalik bazli oneri sunmak.
- Hedef 4: Ciftci odakli, sade ve anlasilir bir mobil arayuz ile kullanim kolayligi saglamak.

## 5) Yenilikci Yonler
- Hafif CNN tabanli yaklasim: ShuffleNet V2 ile mobil uygunlugu on planda tutulmustur.
- Coklu mimari olcek deneyi: 0.5x-2.0x arasi sistematik karsilastirma yapilmistir.
- Gercek zamanli saha kullanimina odak: Cevrimdisi calisma senaryosu hedeflenmistir.
- Karar destek katmani: Teshis sonucuna ek olarak tedavi/yonetim onerisi sunulur.

## 6) Teknik Yontem
### 6.1 Veri ve On Isleme
- Veri kaynagi: Domates yaprak goruntulerinden olusan 10 sinifli veri yapisi.
- Girdi boyutu: 224x224 RGB.
- Normalizasyon: Piksel degerleri [0,1] ve ardindan [-1,1] araligina normalize edilir.
- Veri bolunmesi: Yaklasik %80 egitim - %20 dogrulama/test.

### 6.2 Model Mimarisi
- Omurga: ShuffleNet V2 (0.5x, 1.0x, 1.5x, 2.0x).
- Egitim: 50 epoch, SGD + momentum veya deney senaryosuna gore Adam.
- Amac: En iyi dogruluk ve mobil performans dengesini yakalamak.

### 6.3 Degerlendirme Metrikleri
- Accuracy
- Precision (Weighted/Macro)
- Recall (Weighted/Macro)
- F1-Score (Weighted/Macro)
- Sinif bazli performans ve confusion matrix

## 7) Deneysel Sonuclar (ShuffleNet)
Asagidaki tablo test sonuclarindan derlenmistir.

| Deney | Test Accuracy | Weighted F1 | Transfer Learning | Not |
|---|---:|---:|---|---|
| 1.5x Baseline | 0.997247 | 0.997243 | Kapali | En yuksek test dogrulugu |
| 2.0x Baseline | 0.996145 | 0.996138 | Kapali | Yuksek performans |
| 1.0x Baseline | 0.995595 | 0.995596 | Kapali | Dengeli sonuc |
| 2.0x Aug | 0.992015 | 0.991988 | Kapali | Yuksek genelleme |
| 1.5x Aug | 0.990914 | 0.990897 | Kapali | Yuksek genelleme |
| 1.0x Aug | 0.985132 | 0.985078 | Kapali | Orta-ust seviye |
| 0.5x Aug | 0.982930 | 0.982929 | Kapali | Hafif model |
| 0.5x Baseline | 0.974945 | 0.974927 | Kapali | En hafif mimari |

Gozlem:
- 1.5x baseline en yuksek test performansini vermistir.
- 2.0x mimari cok yakin performans saglarken parametre yukunu artirmaktadir.
- Bazi transfer ogrenme senaryolarinda sinif colapsesi (tek sinifa yigilmaya benzer durum) gorulmustur; bu nedenle mevcut formda ana cozum olarak sifirdan egitim tarafi one cikmaktadir.

## 8) Grafikler ve Sekiller (Rapora Eklenecek)
### 8.1 Uretilen Karsilastirma Grafikleri
1. En iyi 8 deney icin test dogrulugu bar grafigi:
   - Dosya: assets/top8_accuracy_bar.png
2. Secili modellerde validation accuracy epoch egrileri:
   - Dosya: assets/val_accuracy_curves.png
3. Tum deneylerin ozet tablosu (CSV):
   - Dosya: assets/experiment_summary.csv

### 8.2 Mevcut Deney Ciktilari (Dogrudan kullanilabilir)
- 1.5x aug confusion matrix:
  - ../checkpoints_tomato_1_5x_aug/confusion_matrix.png
- 1.5x aug sinif bazli metrik grafigi:
  - ../checkpoints_tomato_1_5x_aug/per_class_metrics.png
- 2.0x aug confusion matrix:
  - ../checkpoints_tomato_2_0x_aug/confusion_matrix.png
- 2.0x aug sinif bazli metrik grafigi:
  - ../checkpoints_tomato_2_0x_aug/per_class_metrics.png

## 9) Mobil Uygulama Entegrasyonu
Planlanan mobil akis:
1. Kamera veya galeriden yaprak goruntusu alimi
2. Goruntu on isleme (boyutlandirma/normalizasyon)
3. On-device model cagrisi ve sinif tahmini
4. Hastalik adi + guven skoru + pratik oneri ekrani

Mobil tarafta beklenen cikti: internet olmadan calisan, tek ekranda tani ve oneri sunan arayuz.

## 10) Is Plani ve Takvim
- WP1: Veri duzenleme ve etiket kalite kontrolu
- WP2: Model egitimi ve hiperparametre optimizasyonu
- WP3: Mobil entegrasyon ve on-device test
- WP4: Saha senaryosu dogrulamasi ve raporlama

Kisa takvim:
- Hafta 1-2: Veri ve deney kurulumu
- Hafta 3-4: Model secimi ve testler
- Hafta 5: Mobil entegrasyon
- Hafta 6: Sunum, rapor ve son kontroller

## 11) Riskler ve Onlemler
- Risk: Siniflar arasi dengesizlik veya asiri ogrenme
  - Onlem: Sinif bazli metrik takibi, erken durdurma, duzenlilestirme
- Risk: Mobilde gecikme/sure uzamasi
  - Onlem: Model olcegi secimi (1.0x/1.5x), niceleme optimizasyonu
- Risk: Transfer ogrenmede performans dususu
  - Onlem: Katman acma stratejisi, dusuk ogrenme hizi ile ince ayar

## 12) Yaygin Etki ve Katma Deger
- Erken teshis ile verim kaybinin azaltilmasi
- Gereksiz ilac kullaniminin azaltilmasi ve cevresel fayda
- Dijital tarim uygulamalarina geciste ciftciye erisilebilir arac
- Yerli ekip tarafindan gelistirilen uygulanabilir teknoloji birikimi

## 13) Sonuc
Proje, domates yaprak hastaliklarinin mobilde ve gercek zamanli teshisine odaklanan, yuksek dogruluklu bir goruntu isleme cozumudur. ShuffleNet tabanli deneylerde %99.72 seviyesine ulasan test dogrulugu elde edilmis, sistemin tarla kosullarinda cevrimdisi kullanimi hedeflenmistir. Bu yapi ile hem teknik basari hem de tarimsal uygulanabilirlik birlikte saglanmaktadir.

---

## 14) Forma Eklenecek Mobil Ekran Goruntusu Istekleri (Yer Tutucu)
Asagidaki ekran goruntuleri rapora eklenmelidir:
1. Ana ekran: Kamera ac / galeriden sec butonlari, sade kullanim akisi.
2. Tani sonucu ekrani: Tahmin edilen hastalik adi, guven yuzdesi, en yakin 2 alternatif sinif.
3. Oneri ekrani: Hastaliga ozel mudahale adimlari (kisa madde listesi).
4. Gecmis kayit ekrani: Tarih, tahmin, konum veya not bilgisi (varsa).

Not: Eger ekran goruntuleri henuz hazir degilse, bu bolume her ekran icin yukaridaki aciklamalar aynen yazilip "Temsili ekran eklenecek" notu dusulebilir.
