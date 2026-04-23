# Phase 5 - Threshold Calibration Automation

Bu dokuman, Faz 5 / Adim 2 kapsaminda karar esiklerini negatif orneklerle kalibre etmek icin tek komut akisini tarif eder.

## Komut

Proje kokunden:

```bash
cd tomatech-android
./scripts/run_phase5_threshold_calibration.sh
```

Opsiyonel olarak negatif klasor yolu verilebilir:

```bash
./scripts/run_phase5_threshold_calibration.sh /custom/negative_dir
```

Ek argumanlar direkt kalibrasyon scriptine iletilir. Ornek (hizli smoke):

```bash
./scripts/run_phase5_threshold_calibration.sh \
  ../shufflenet-v2-tensorflow/calibration_data/negatives \
  --max-positive-samples 400 \
  --max-negative-samples 200
```

## Uretilen Ciktilar

- JSON rapor: shufflenet-v2-tensorflow/artifacts/tflite/threshold_calibration_report_phase5_<timestamp>.json
- CSV adaylar: shufflenet-v2-tensorflow/artifacts/tflite/threshold_calibration_candidates_phase5_<timestamp>.csv
- Terminalde Android icin oneri snippet'i

## App'te Beklenen Etki

- Yaprak disi goruntulerin Geçersiz Goruntu durumuna daha tutarli yonlenmesi.
- Yaprakli/pozitif goruntulerde gereksiz Geçersiz artisini kontrol altinda tutma.
- Belirsiz ve Geçersiz ayriminin daha stabil hale gelmesi.

## Uygulama Adimi

Script cikisindaki onerilen 3 esik degerini su dosyaya uygula:
- app/src/main/java/com/tomatech/mobile/DecisionThresholds.kt

Sonra:

```bash
JAVA_HOME=/home/exc/.local/share/JetBrains/Toolbox/apps/android-studio/jbr \
PATH="$JAVA_HOME/bin:$PATH" \
./gradlew :app:assembleDebug
```

ve cihazda kisa regresyon retesti yap.

## Son Kalibrasyon Notu (2026-04-19)

- Expanded negatif set (369 gorsel) ile tam kosu sonucu onerilen esik:
  - INVALID_IMAGE_CONFIDENCE_THRESHOLD = 0.70f
  - CONFIDENT_DIAGNOSIS_THRESHOLD = 0.90f
  - MIN_MARGIN_THRESHOLD = 0.10f
- Rapor: shufflenet-v2-tensorflow/artifacts/tflite/threshold_calibration_report_phase5_20260419_143814.json
