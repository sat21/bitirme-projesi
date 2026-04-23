# TomaTech Android MVP

Native Android (Kotlin + Compose) offline tomato disease diagnosis app.

## Features (MVP)

- Gallery image selection
- CameraX live preview capture (tap-to-focus + flash toggle + front/back lens switch)
- On-device TensorFlow Lite inference
- Three-state decision layer (Diagnosis / Uncertain / Invalid Image)
- Result safety layer with status-based next-step guidance and invalid-image suppression
- Top-1 and Top-3 prediction output
- Inference latency display (ms)

## Project Structure

- app/src/main/java/com/tomatech/mobile/MainActivity.kt: app entry and screen host
- app/src/main/java/com/tomatech/mobile/TomatoViewModel.kt: state + inference orchestration
- app/src/main/java/com/tomatech/mobile/DecisionThresholds.kt: uc-durum karar esikleri
- app/src/main/java/com/tomatech/mobile/ui/screens/TomatoDiagnosisScreen.kt: ana ekran akisi
- app/src/main/java/com/tomatech/mobile/ui/components/: yeniden kullanilabilir Compose bilesenleri
- app/src/main/java/com/tomatech/mobile/ml/TomatoClassifier.kt: TFLite interpreter wrapper
- app/src/main/java/com/tomatech/mobile/ml/ImagePreprocessor.kt: 224x224 RGB + [-1,1] normalization
- app/src/main/assets/checkpoints_tomato_1_5x_baseline_best_model_int8.tflite: model
- app/src/main/assets/labels_tomato_10.txt: labels

## Requirements

- Android Studio Iguana or newer
- Android SDK 34
- Java 17

## Run

1. Open tomatech-android folder in Android Studio.
2. Let Gradle sync complete.
3. Select an emulator/device with Android 9+ (API 28+).
4. Run the app.

## Notes

- Model is the INT8 variant exported from this workspace.
- Preprocess contract is aligned with Python:
  - Resize 224x224
  - Normalize with x / 127.5 - 1.0
- Current latency is measured on device at inference call level and shown in UI.
- Three-state decision thresholds (in DecisionThresholds.kt):
  - Invalid Image: top-1 confidence < 0.70
  - Uncertain: top-1 confidence < 0.90 OR top-1/top-2 margin < 0.10
  - Diagnosis: otherwise
- Device-level validation checklist: docs/device_test_checklist.md
- QA closure report: docs/qa_closure_report.md
- Faz 5 esik kalibrasyon otomasyonu: docs/phase5_threshold_calibration.md

## Next Steps

- Add CameraX UX polish (AE lock and focus-state hint).
- Add offline diagnosis history (Room).
- Calibrate thresholds with real negative samples (non-leaf and non-tomato images).

## Phase 5 Quick Run

```bash
cd tomatech-android
./scripts/run_phase5_threshold_calibration.sh
```

Bu komut, negatif ornekler ile karar esiklerini kalibre eder ve Android icin onerilen esik snippet'ini terminale yazar.
