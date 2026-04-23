# Android Offline Deployment - Faz 1

Bu klasor, ShuffleNetV2 domates modelini Android icin TFLite formatina donusturmek ve Keras-TFLite ciktilarini karsilastirmak icin kullanilir.

## Hedef

- .keras modelini TFLite formatina export etmek
- Model boyutunu olcmek
- Keras ve TFLite accuracy farkini hesaplamak
- TFLite latency degerlerini raporlamak

## Dosyalar

- export_tflite.py: Keras -> TFLite donusumu (fp32/fp16/int8)
- validate_tflite.py: Keras ve TFLite model tahminlerini ayni test splitinde karsilastirma
- compare_single_image.py: Tek goruntude Keras-TFLite tahminlerini yan yana karsilastirma
- calibrate_decision_thresholds.py: Uc-durum karar esiklerini (DIAGNOSIS/UNCERTAIN/INVALID_IMAGE) kalibre etme

## On Kosul

- TensorFlow, NumPy, Pillow kurulu olmali
- Domates veri seti yolu varsayilan olarak su sekilde beklenir:
  - /mnt/021630F41630E9F5/PROJECTS/torch/tomato
- Varsayilan model:
  - shufflenet-v2-tensorflow/checkpoints_tomato_1_5x_baseline/best_model.keras

## 1) FP16 TFLite Export

Shufflenet-v2-tensorflow klasorunde calistir:

```bash
python deployment/export_tflite.py \
  --quantization fp16
```

Olusan ciktilar:

- artifacts/tflite/checkpoints_tomato_1_5x_baseline_best_model_fp16.tflite
- artifacts/tflite/labels_tomato_10.txt
- artifacts/tflite/checkpoints_tomato_1_5x_baseline_best_model_fp16_metadata.json

## 2) INT8 TFLite Export (Opsiyonel)

```bash
python deployment/export_tflite.py \
  --quantization int8 \
  --num-representative-samples 256
```

Not:

- INT8 donusum representative dataset kullanir.
- Tam INT8 giris/cikis isterseniz --enforce-full-int8 parametresini ekleyin.

## 3) Keras vs TFLite Dogrulama

```bash
python deployment/validate_tflite.py \
  --tflite-model artifacts/tflite/checkpoints_tomato_1_5x_baseline_best_model_fp16.tflite \
  --max-test-samples 1000
```

Olusan raporlar:

- artifacts/tflite/validation_summary.csv
- artifacts/tflite/validation_mismatches.csv

## 4) Tek Goruntu Karsilastirma (Keras vs TFLite)

Belirli bir goruntu icin iki modelin yan yana sonucunu gorebilirsiniz.

Yerel dosya ile:

```bash
python deployment/compare_single_image.py \
  --image-path "/tam/yol/goruntu.jpg" \
  --top-k 3 \
  --output-json artifacts/tflite/single_image_compare.json
```

URL ile:

```bash
python deployment/compare_single_image.py \
  --image-url "https://.../image.jpg" \
  --top-k 3 \
  --output-json artifacts/tflite/single_image_compare_url.json
```

Script su bilgileri verir:

- Keras Top-K sinif ve guven degerleri
- TFLite Top-K sinif ve guven degerleri
- Uygulamadaki karar katmanina gore durum (DIAGNOSIS / UNCERTAIN / INVALID_IMAGE)
- Top-1 ayni mi ve olasilik dagilimi farki (L1 mesafesi)

## 5) Karar Esigi Kalibrasyonu (Hard-Negative ile)

Bu adim, uygulamadaki uc-durum karar katmani icin daha iyi esik secmeye yarar.

- Pozitif set: `tomato/` altindaki sinifli veri
- Negatif set: domates disi goruntulerin oldugu klasor (hazir klasor: `calibration_data/negatives/`)

Komut:

```bash
python deployment/calibrate_decision_thresholds.py \
  --negative-dir /tam/yol/negatives \
  --report-json artifacts/tflite/threshold_calibration_report.json \
  --report-csv artifacts/tflite/threshold_calibration_candidates.csv
```

Script ciktisi:

- Mevcut esiklerin performansi
- Onerilen yeni esikler
- En iyi aday kombinasyonlarin CSV raporu
- Android koduna yapistirilabilir `DecisionThresholds` Kotlin snippet'i

Not:

- `--negative-dir` vermezseniz script sadece pozitif setle calisir; bu durumda "gecersiz goruntu" kalibrasyonu zayif olur.
- En dogru sonuc icin negatif sete sahada karsilasabileceginiz goruntuleri ekleyin (insan, masa, duvar, diger bitkiler, farkli yapraklar vb.).

## Kabul Kriterleri (Faz 1)

- Accuracy farki <= 0.01 (mutlak)
- Model boyutu < 20 MB
- Ortalama tek-girdi latency hedefi < 500 ms

## Faz 2 Hazirlik

Faz 1 basarili olduktan sonra Android tarafa gecis:

- TensorFlow Lite Interpreter entegrasyonu
- CameraX ile goruntu alma
- Ayni preprocess adimlari (224x224, RGB, normalize [-1,1])
- Sonuc ekrani: Top-1 sinif + guven skoru
