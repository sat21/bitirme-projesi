# Phase 6 - Temperature Calibration

Bu dokuman, modelin asiri guvenli (%99-%100) cikti egilimini azaltmak icin temperature scaling kalibrasyonunu tarif eder.

## Neden?

Softmax olasiliklari, ozellikle internetten gelen OOD (out-of-distribution) goruntulerde asiri guvenli olabilir.
Temperature scaling, sinif siralamasini bozmadan olasilik dagilimini yumusatir.

- Dusuk temperature (<1.0): daha sivri ve daha guvenli cikti
- Yuksek temperature (>1.0): daha yumusak ve daha temkinli cikti

## Komut

Proje kokunden:

```bash
cd tomatech-android
./scripts/run_phase6_temperature_calibration.sh
```

Opsiyonel olarak negatif klasor verilebilir:

```bash
./scripts/run_phase6_temperature_calibration.sh /custom/negative_dir
```

Ek argumanlar dogrudan calibration scriptine iletilir. Ornek:

```bash
./scripts/run_phase6_temperature_calibration.sh \
  ../shufflenet-v2-tensorflow/calibration_data/negatives_phase5_expanded_20260419 \
  --max-positive-samples 1500 \
  --temperature-min 0.90 \
  --temperature-max 3.20 \
  --temperature-step 0.10
```

## Uretilen Ciktilar

- JSON rapor: shufflenet-v2-tensorflow/artifacts/tflite/temperature_calibration_report_phase6_<timestamp>.json
- CSV adaylar: shufflenet-v2-tensorflow/artifacts/tflite/temperature_calibration_candidates_phase6_<timestamp>.csv
- Terminalde Android icin ModelCalibration snippet'i

## Guard-Rail Secim Mantigi

Calibration scripti sadece objective minimizasyonu yapmaz, ayni zamanda su kalite kosullarini da zorunlu tutar:

- min_pos_diag_rate >= 0.94
- min_pos_diag_acc >= 0.98
- max_pos_invalid_rate <= 0.03

Bu sayede temperature degeri negatifleri baskilarken pozitif tanilari asiri dusurmez.

## Android'e Uygulama

Script cikisindaki onerilen degeri su dosyaya uygula:

- app/src/main/java/com/tomatech/mobile/ml/ModelCalibration.kt

Sonra:

```bash
JAVA_HOME=/home/exc/.local/share/JetBrains/Toolbox/apps/android-studio/jbr \
PATH="$JAVA_HOME/bin:$PATH" \
./gradlew :app:testDebugUnitTest :app:assembleDebug
```

ve cihazda kisa retest yap:

- Yaprakli saglikli
- Yaprakli hastalikli
- Yaprak disi (insan, masa, ekran)

## Son Kalibrasyon Notu (2026-04-27)

- Guard-railli sweep sonucu onerilen deger: TEMPERATURE_SCALING_FACTOR = 2.40f
- Rapor: shufflenet-v2-tensorflow/artifacts/tflite/temperature_calibration_report_phase6_guardrail_20260427.json
- Adaylar: shufflenet-v2-tensorflow/artifacts/tflite/temperature_calibration_candidates_phase6_guardrail_20260427.csv
