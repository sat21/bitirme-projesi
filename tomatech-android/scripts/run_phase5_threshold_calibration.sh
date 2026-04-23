#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANDROID_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="$(cd "$ANDROID_DIR/.." && pwd)"
SHUFFLE_DIR="$WORKSPACE_ROOT/shufflenet-v2-tensorflow"
CALIBRATE_SCRIPT="$SHUFFLE_DIR/deployment/calibrate_decision_thresholds.py"

if [[ ! -f "$CALIBRATE_SCRIPT" ]]; then
  echo "[ERROR] Calibration script not found: $CALIBRATE_SCRIPT" >&2
  exit 1
fi

if [[ -x "$WORKSPACE_ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$WORKSPACE_ROOT/.venv/bin/python"
elif [[ -x "$WORKSPACE_ROOT/.venv-1/bin/python" ]]; then
  PYTHON_BIN="$WORKSPACE_ROOT/.venv-1/bin/python"
else
  PYTHON_BIN="python3"
fi

NEGATIVE_DIR_DEFAULT="$SHUFFLE_DIR/calibration_data/negatives"
NEGATIVE_DIR="${1:-$NEGATIVE_DIR_DEFAULT}"

if [[ ! -d "$NEGATIVE_DIR" ]]; then
  echo "[ERROR] Negative directory not found: $NEGATIVE_DIR" >&2
  exit 1
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
REPORT_JSON="$SHUFFLE_DIR/artifacts/tflite/threshold_calibration_report_phase5_${TIMESTAMP}.json"
REPORT_CSV="$SHUFFLE_DIR/artifacts/tflite/threshold_calibration_candidates_phase5_${TIMESTAMP}.csv"

echo "[INFO] Python: $PYTHON_BIN"
echo "[INFO] Negative dir: $NEGATIVE_DIR"
echo "[INFO] Report JSON: $REPORT_JSON"
echo "[INFO] Report CSV: $REPORT_CSV"

"$PYTHON_BIN" "$CALIBRATE_SCRIPT" \
  --negative-dir "$NEGATIVE_DIR" \
  --report-json "$REPORT_JSON" \
  --report-csv "$REPORT_CSV" \
  "${@:2}"

echo "[INFO] Parsing recommendation..."
"$PYTHON_BIN" - <<'PY' "$REPORT_JSON"
import json
import pathlib
import sys

report_path = pathlib.Path(sys.argv[1])
report = json.loads(report_path.read_text(encoding="utf-8"))
rec = report["recommended_thresholds"]
metrics = rec["metrics"]

print("\n[RECOMMENDED FOR ANDROID]")
print("object DecisionThresholds {")
print(f"    const val INVALID_IMAGE_CONFIDENCE_THRESHOLD = {rec['invalid_conf']:.2f}f")
print(f"    const val CONFIDENT_DIAGNOSIS_THRESHOLD = {rec['confident_conf']:.2f}f")
print(f"    const val MIN_MARGIN_THRESHOLD = {rec['margin']:.2f}f")
print("}")
print("\n[QUALITY SNAPSHOT]")
print(f"positive_diag_rate={metrics['pos_diag_rate']:.4f}")
print(f"positive_invalid_rate={metrics['pos_invalid_rate']:.4f}")
print(f"negative_diag_rate={metrics['neg_diag_rate']:.4f}")
print(f"negative_invalid_rate={metrics['neg_invalid_rate']:.4f}")
print(f"score={metrics['score']:.4f}")
print(f"json_report={report_path}")
print(f"csv_candidates={report['top_candidates_csv']}")
PY

echo "[DONE] Phase-5 threshold calibration completed."
