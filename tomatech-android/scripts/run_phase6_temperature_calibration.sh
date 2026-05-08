#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SHUFFLE_DIR="$WORKSPACE_ROOT/shufflenet-v2-tensorflow"
CALIBRATE_SCRIPT="$SHUFFLE_DIR/deployment/calibrate_temperature.py"

if [[ ! -f "$CALIBRATE_SCRIPT" ]]; then
  echo "[ERROR] Temperature calibration script not found: $CALIBRATE_SCRIPT" >&2
  exit 1
fi

if [[ -x "$WORKSPACE_ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$WORKSPACE_ROOT/.venv/bin/python"
elif [[ -x "$WORKSPACE_ROOT/.venv-1/bin/python" ]]; then
  PYTHON_BIN="$WORKSPACE_ROOT/.venv-1/bin/python"
else
  PYTHON_BIN="python3"
fi

NEGATIVE_DIR_DEFAULT="$SHUFFLE_DIR/calibration_data/negatives_phase5_expanded_20260419"
NEGATIVE_DIR="${1:-$NEGATIVE_DIR_DEFAULT}"

if [[ ! -d "$NEGATIVE_DIR" ]]; then
  echo "[ERROR] Negative directory not found: $NEGATIVE_DIR" >&2
  exit 1
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
REPORT_JSON="$SHUFFLE_DIR/artifacts/tflite/temperature_calibration_report_phase6_${TIMESTAMP}.json"
REPORT_CSV="$SHUFFLE_DIR/artifacts/tflite/temperature_calibration_candidates_phase6_${TIMESTAMP}.csv"

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
rec = report["recommended_temperature"]

print("\n[RECOMMENDED FOR ANDROID]")
print("object ModelCalibration {")
print("    // Recommended by phase-6 temperature calibration")
print(f"    const val TEMPERATURE_SCALING_FACTOR = {rec['temperature']:.2f}f")
print("}")

print("\n[QUALITY SNAPSHOT]")
print(f"objective={rec['objective']:.6f}")
print(f"pos_diag_rate={rec['eval_pos_diag_rate']:.4f}")
print(f"pos_invalid_rate={rec['eval_pos_invalid_rate']:.4f}")
print(f"pos_diag_acc={rec['eval_pos_diag_acc']:.4f}")
if "eval_neg_diag_rate" in rec:
    print(f"neg_diag_rate={rec['eval_neg_diag_rate']:.4f}")
if "eval_neg_invalid_rate" in rec:
    print(f"neg_invalid_rate={rec['eval_neg_invalid_rate']:.4f}")
print(f"json_report={report_path}")
print(f"csv_candidates={report['top_candidates_csv']}")
PY

echo "[DONE] Phase-6 temperature calibration completed."
