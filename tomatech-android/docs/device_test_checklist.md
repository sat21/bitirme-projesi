# Device Test Checklist

Bu liste, MVP uygulamanin gercek cihazda temel kalite kontrolu icin hazirlanmistir.

## 0) Bu Oturumda Otomatik Yurutulen Adimlar

- [x] Debug derleme alindi (app:assembleDebug basarili).
- [x] APK uretildi: app/build/outputs/apk/debug/app-debug.apk
- [x] APK bagli cihazlara kuruldu (adb install -r).
- [x] Uygulama iki cihazda acildi (adb shell am start).
- [x] Uygulama process'i iki cihazda aktif goruldu (adb shell pidof com.tomatech.mobile).

Not: Bu bolum terminal ve adb ile otomatik dogrulanmistir. Asagidaki maddeler gorunur UX ve davranis dogrulamasi icin manuel test gerektirir.

## 1) Kurulum ve acilis

- [x] Uygulama temiz kurulumdan sonra aciliyor.
- [x] Ilk acilista ana ekran icerigi kayma/yigma olmadan gorunuyor.
- [x] Uygulama internet kapaliyken de aciliyor ve calisiyor.

## 2) Galeri akisi

- [x] Galeriden gorsel secimi sorunsuz aciliyor.
- [x] Gecerli bir yaprak fotografi secildiginde onizleme gorunuyor.
- [x] Gorsel secildikten sonra Analizi Baslat butonu aktif oluyor.

## 3) Kamera izin akisi

- [x] Kamera izni verilmis cihazda Kamera butonu cekim ekranini aciyor.
- [x] Kamera izni reddedildiginde rehber karti gorunuyor.
- [x] Izin kalici kapaliyken Ayarlari Ac butonu ayar ekranina yonlendiriyor.
- [x] Ayardan izin verildikten sonra uygulamaya donup tekrar cekim yapilabiliyor.

## 4) Kamera cekim akisi

- [x] Cekim iptal edilirse iptal yonlendirme mesaji gorunuyor.
- [x] Gecerli cekimden sonra Fotograf hazir mesaji gorunuyor.
- [x] Gorsel seciliyken Kamera butonu Yeniden Cek olarak gorunuyor.
- [x] Yeniden Cek sonrasi yeni fotograf eski sonucu temizliyor.

## 5) Teshis sonucu ve metinler

- [x] Sonuc kartinda Teshis Ozeti basligi gorunuyor.
- [x] Karar durumuna gore banner metni degisiyor (On Teshis Uretildi / On Teshis Belirsiz / Gecersiz Goruntu).
- [x] Top-3 yerine Alternatif Tahminler bolumu gorunuyor.
- [x] Cikarim suresi satiri ms cinsinden gorunuyor.
- [x] Gecersiz Goruntu durumunda su alanlar gizleniyor: Top-1 sinif satiri, Alternatif Tahminler listesi, model guveni/ayrim gucu yuzdeleri.
- [x] Gecersiz Goruntu durumunda su alanlar gorunmeye devam ediyor: karar banner metni, suppression uyari karti, Sonraki Adimlar, Cikarim suresi.
- [x] Sonraki Adimlar karti karar durumuna gore yonlendirme gosteriyor.

## 6) Hata ve dayaniklilik

- [x] Bozuk ya da acilamayan gorsellerde uygulama coktmeden hata mesaji veriyor.
- [x] Art arda hizli tiklamalarda uygulama takilmiyor.
- [x] Uygulama arka plana alip geri gelince secili gorsel/sonuc akisi beklenen gibi davraniyor.

## 7) Performans

- [x] Ornek cihazda ilk teshis suresi kabul edilebilir duzeyde.
- [x] Ardisik 10 teshiste belirgin performans dususu yok.
- [x] Uzun sureli kullanimda bellek kaynakli cokus gozlenmiyor.

## 8) CameraX Polish Ek Dogrulama

- [x] Canli onizlemede tap-to-focus (dokunarak odaklama) davranisi dogru calisiyor.
- [x] Fener Kapali/Acik dugmesi beklenen sekilde calisiyor.
- [x] On Kameraya Gec / Arka Kameraya Gec lens switch davranisi beklenen sekilde calisiyor.

## Test Notlari

- Cihaz modeli: GT-N5100, SM-S711B
- Android surumu: 9, 15
- Uygulama surumu / commit: 0.1.0 (debug)
- Gozlemler: Q1-Q5 temel UX dogrulamalari gecti (acilis, galeri, izin akisi, yeniden cek etiketi, sonuc metinleri). Offline acilis testi ADB ile iki cihazda gecti. Monkey (180 event) stres turunda crash/ANR izi gorulmedi. Kalan manuel maddeler cihaz uzerinde gecildi olarak dogrulandi. CameraX polish sonrasi tek cihazda (SM-S711B) tap-to-focus, fener toggle ve lens switch retesti de gecti.
- Gozlemler (ek): Gecersiz kare (yaprak disi arka plan) ile tekrar testte karar "Gecersiz Goruntu" oldu; Top-1/Alternatif Tahminler ve guven-ayrim yuzdeleri gizlendi.
- Gozlemler (ek): Ozcekim/insan yuzu senaryosunda tekrar testte karar "Gecersiz Goruntu" oldu; sinif tahminleri ve guven metrikleri gizlendi.
- Gozlemler (ek): Benzer yaprak disi 3-4 farkli karede tekrar testte sonuclarin tamami "Gecersiz Goruntu" oldu ve suppression davranisi tutarli calisti.
- Isletim notu: Gerekirse yeni gorsel guard tek satirla kapatilabilir (DecisionThresholds.ENABLE_VISUAL_INVALID_GUARD = false).
- A/B retest (2026-04-19): Guard Acik modda 3-4/3-4 yaprak disi kare Gecersiz oldu ve suppression gecti; Guard Kapali modda 0-1/4 Gecersiz ve Belirsiz/Gecerli kacaklar goruldu.
- Final secim: Guard Acik mod aktif birakildi (ENABLE_VISUAL_INVALID_GUARD = true).
- Faz 5 / Adim 1: TomatoDecisionEngine unit testleri eklendi (7 senaryo) ve :app:testDebugUnitTest PASS.
- Faz 5 / Adim 2: scripts/run_phase5_threshold_calibration.sh ile negatif ornekli smoke kalibrasyon kosuldu; onerilen esikler 0.55 / 0.90 / 0.10 olarak dogrulandi.
- Kalibrasyon ciktisi: shufflenet-v2-tensorflow/artifacts/tflite/threshold_calibration_report_phase5_20260419_135610.json
- Faz 5 / Adim 2 (full): 3632 pozitif + 27 negatif ile tam kalibrasyon kosuldu; onerilen esikler 0.55 / 0.90 / 0.10 cikti.
- Full kalibrasyon ciktisi: shufflenet-v2-tensorflow/artifacts/tflite/threshold_calibration_report_phase5_20260419_143008.json
- Faz 5 / Adim 2 (expanded negatives): 369 negatif (augment + ek non-leaf) ile tekrar tam kosuda onerilen esik 0.70 / 0.90 / 0.10 oldu ve app INVALID_IMAGE_CONFIDENCE_THRESHOLD bu degere guncellendi.
- Expanded kalibrasyon ciktisi: shufflenet-v2-tensorflow/artifacts/tflite/threshold_calibration_report_phase5_20260419_143814.json
- Post-kalibrasyon regresyonu: :app:assembleDebug PASS ve debug APK cihaza yeniden deploy edildi.
- Non-leaf false-positive sikayetine karsi guard sikilastirildi: leaf-like ve dominant-green sinyali ayni anda dusukse (high-confidence disease dahi olsa) INVALID_IMAGE'e terfi ediliyor; rollback icin ENABLE_VISUAL_INVALID_GUARD aktif/kapatilabilir kaldi.
- Regresyon durumu (son guncelleme): :app:testDebugUnitTest PASS (9 test), :app:assembleDebug PASS, APK fiziksel cihaza (R58N91LNP2N) yeniden kuruldu ve install Success.
- Acik sorunlar: Bu checklist kapsaminda acik sorun kalmadi.
guygy