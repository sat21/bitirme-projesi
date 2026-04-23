# Hard-Negative Images

Bu klasore domates disi veya modeli sasirtabilecek goruntuleri koyun.

Ornekler:
- Insan, yuz, el, kiyafet
- Masa, sandalye, duvar, zemin
- Diger bitki yapraklari
- Cok uzak veya asiri bulanik yaprak
- Asiri karanlik / asiri parlak cekimler

Desteklenen formatlar:
- .jpg
- .jpeg
- .png

Kalibrasyon komutu:

```bash
python deployment/calibrate_decision_thresholds.py \
  --negative-dir calibration_data/negatives \
  --report-json artifacts/tflite/threshold_calibration_report_with_negatives.json \
  --report-csv artifacts/tflite/threshold_calibration_candidates_with_negatives.csv
```
