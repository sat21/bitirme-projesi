#!/usr/bin/env python3
"""Calibrate three-state decision thresholds for mobile inference.

States:
- DIAGNOSIS
- UNCERTAIN
- INVALID_IMAGE
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import tensorflow as tf

CLASS_NAMES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

CURRENT_INVALID_THRESHOLD = 0.50
CURRENT_CONFIDENT_THRESHOLD = 0.75
CURRENT_MARGIN_THRESHOLD = 0.20


@dataclass
class Thresholds:
    invalid_conf: float
    confident_conf: float
    margin: float


@dataclass
class Metrics:
    score: float
    positive_count: int
    negative_count: int
    pos_diag_rate: float
    pos_uncertain_rate: float
    pos_invalid_rate: float
    pos_overall_acc: float
    pos_diag_acc: float
    neg_diag_rate: Optional[float]
    neg_uncertain_rate: Optional[float]
    neg_invalid_rate: Optional[float]


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    default_tflite = project_root / "artifacts" / "tflite" / "checkpoints_tomato_1_5x_baseline_best_model_int8.tflite"
    default_data = project_root.parent / "tomato"
    default_report_json = project_root / "artifacts" / "tflite" / "threshold_calibration_report.json"
    default_report_csv = project_root / "artifacts" / "tflite" / "threshold_calibration_candidates.csv"

    parser = argparse.ArgumentParser(description="Karar esiklerini pozitif/negatif goruntu seti ile kalibre eder.")
    parser.add_argument("--tflite-model", type=Path, default=default_tflite)
    parser.add_argument("--data-dir", type=Path, default=default_data)
    parser.add_argument("--negative-dir", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-positive-samples", type=int, default=0)
    parser.add_argument("--max-negative-samples", type=int, default=0)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--invalid-min", type=float, default=0.35)
    parser.add_argument("--invalid-max", type=float, default=0.70)
    parser.add_argument("--invalid-step", type=float, default=0.05)
    parser.add_argument("--confident-min", type=float, default=0.65)
    parser.add_argument("--confident-max", type=float, default=0.90)
    parser.add_argument("--confident-step", type=float, default=0.05)
    parser.add_argument("--margin-min", type=float, default=0.10)
    parser.add_argument("--margin-max", type=float, default=0.35)
    parser.add_argument("--margin-step", type=float, default=0.05)
    parser.add_argument("--top-candidates", type=int, default=20)
    parser.add_argument("--report-json", type=Path, default=default_report_json)
    parser.add_argument("--report-csv", type=Path, default=default_report_csv)
    return parser.parse_args()


def collect_positive_dataset(data_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    image_paths: List[str] = []
    labels: List[int] = []
    patterns = ("*.jpg", "*.JPG", "*.jpeg", "*.png")

    for idx, class_name in enumerate(CLASS_NAMES):
        class_dir = data_dir / class_name
        if not class_dir.exists():
            continue

        files: List[Path] = []
        for pattern in patterns:
            files.extend(sorted(class_dir.glob(pattern)))

        image_paths.extend(str(path) for path in files)
        labels.extend([idx] * len(files))

    if not image_paths:
        raise ValueError(f"Pozitif veri seti bulunamadi: {data_dir}")

    return np.array(image_paths), np.array(labels, dtype=np.int32)


def collect_negative_dataset(negative_dir: Path) -> np.ndarray:
    patterns = ("*.jpg", "*.JPG", "*.jpeg", "*.png")
    files: List[Path] = []

    for pattern in patterns:
        files.extend(sorted(negative_dir.rglob(pattern)))

    if not files:
        raise ValueError(f"Negative veri seti bulunamadi: {negative_dir}")

    return np.array([str(path) for path in files])


def split_test_set(
    image_paths: np.ndarray,
    labels: np.ndarray,
    train_split: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(image_paths))
    split_idx = int(len(image_paths) * train_split)
    test_idx = indices[split_idx:]
    return image_paths[test_idx], labels[test_idx]


def preprocess_image(image_path: Path, image_size: int) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    img = img.resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return np.expand_dims(arr, axis=0).astype(np.float32)


def build_interpreter(tflite_path: Path, num_threads: int):
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path), num_threads=num_threads)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    return interpreter, input_detail, output_detail


def quantize_input(input_tensor: np.ndarray, input_detail: dict) -> np.ndarray:
    dtype = input_detail["dtype"]
    if dtype == np.float32:
        return input_tensor.astype(np.float32)

    scale, zero_point = input_detail["quantization"]
    if scale == 0:
        raise ValueError("Input quantization scale 0 olamaz.")

    quantized = np.round(input_tensor / scale + zero_point)
    if dtype == np.int8:
        quantized = np.clip(quantized, -128, 127)
    elif dtype == np.uint8:
        quantized = np.clip(quantized, 0, 255)

    return quantized.astype(dtype)


def dequantize_output(output_tensor: np.ndarray, output_detail: dict) -> np.ndarray:
    dtype = output_detail["dtype"]
    if dtype == np.float32:
        return output_tensor.astype(np.float32)

    scale, zero_point = output_detail["quantization"]
    if scale == 0:
        return output_tensor.astype(np.float32)

    return (output_tensor.astype(np.float32) - zero_point) * scale


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float64)
    logits = logits - np.max(logits)
    e = np.exp(logits)
    s = np.sum(e)
    if s == 0:
        return np.zeros_like(logits, dtype=np.float64)
    return e / s


def ensure_probabilities(output_vector: np.ndarray) -> np.ndarray:
    if np.all(output_vector >= 0.0) and np.all(output_vector <= 1.0) and np.isclose(np.sum(output_vector), 1.0, atol=1e-3):
        return output_vector.astype(np.float64)
    return softmax(output_vector)


def predict_probabilities(
    interpreter,
    input_detail: dict,
    output_detail: dict,
    image_paths: np.ndarray,
    image_size: int,
) -> np.ndarray:
    probs: List[np.ndarray] = []

    for path in image_paths:
        image_batch = preprocess_image(Path(path), image_size)
        model_input = quantize_input(image_batch, input_detail)

        interpreter.set_tensor(input_detail["index"], model_input)
        interpreter.invoke()

        raw_output = interpreter.get_tensor(output_detail["index"])[0]
        output = dequantize_output(raw_output, output_detail)
        probs.append(ensure_probabilities(output))

    return np.vstack(probs)


def to_features(probabilities: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    top_indices = np.argsort(probabilities, axis=1)[:, ::-1]
    top1_idx = top_indices[:, 0]
    top2_idx = top_indices[:, 1]

    row_ids = np.arange(len(probabilities))
    top1_conf = probabilities[row_ids, top1_idx]
    top2_conf = probabilities[row_ids, top2_idx]
    margin = top1_conf - top2_conf

    return top1_idx, top1_conf, margin


def classify_state(top1_conf: np.ndarray, margin: np.ndarray, th: Thresholds) -> np.ndarray:
    states = np.full(len(top1_conf), "DIAGNOSIS", dtype=object)
    invalid_mask = top1_conf < th.invalid_conf
    uncertain_mask = (~invalid_mask) & ((top1_conf < th.confident_conf) | (margin < th.margin))

    states[invalid_mask] = "INVALID_IMAGE"
    states[uncertain_mask] = "UNCERTAIN"
    return states


def evaluate_thresholds(
    th: Thresholds,
    pos_top1_idx: np.ndarray,
    pos_top1_conf: np.ndarray,
    pos_margin: np.ndarray,
    pos_labels: np.ndarray,
    neg_top1_conf: Optional[np.ndarray],
    neg_margin: Optional[np.ndarray],
) -> Metrics:
    pos_states = classify_state(pos_top1_conf, pos_margin, th)

    pos_diag_mask = pos_states == "DIAGNOSIS"
    pos_uncertain_mask = pos_states == "UNCERTAIN"
    pos_invalid_mask = pos_states == "INVALID_IMAGE"

    pos_pred_correct = (pos_top1_idx == pos_labels)
    pos_overall_acc = float(np.mean(pos_pred_correct))

    if np.any(pos_diag_mask):
        pos_diag_acc = float(np.mean(pos_pred_correct[pos_diag_mask]))
    else:
        pos_diag_acc = 0.0

    pos_diag_rate = float(np.mean(pos_diag_mask))
    pos_uncertain_rate = float(np.mean(pos_uncertain_mask))
    pos_invalid_rate = float(np.mean(pos_invalid_mask))

    if neg_top1_conf is None or neg_margin is None:
        neg_diag_rate = None
        neg_uncertain_rate = None
        neg_invalid_rate = None
        score = (
            3.0 * pos_diag_rate
            + 1.5 * pos_diag_acc
            - 3.0 * pos_invalid_rate
        )
        negative_count = 0
    else:
        neg_states = classify_state(neg_top1_conf, neg_margin, th)
        neg_diag_rate = float(np.mean(neg_states == "DIAGNOSIS"))
        neg_uncertain_rate = float(np.mean(neg_states == "UNCERTAIN"))
        neg_invalid_rate = float(np.mean(neg_states == "INVALID_IMAGE"))

        score = (
            4.0 * neg_invalid_rate
            + 2.0 * neg_uncertain_rate
            - 8.0 * neg_diag_rate
            + 3.0 * pos_diag_rate
            + 1.5 * pos_diag_acc
            - 3.0 * pos_invalid_rate
        )
        negative_count = len(neg_top1_conf)

    return Metrics(
        score=score,
        positive_count=len(pos_labels),
        negative_count=negative_count,
        pos_diag_rate=pos_diag_rate,
        pos_uncertain_rate=pos_uncertain_rate,
        pos_invalid_rate=pos_invalid_rate,
        pos_overall_acc=pos_overall_acc,
        pos_diag_acc=pos_diag_acc,
        neg_diag_rate=neg_diag_rate,
        neg_uncertain_rate=neg_uncertain_rate,
        neg_invalid_rate=neg_invalid_rate,
    )


def frange(start: float, stop: float, step: float) -> List[float]:
    values = []
    current = start
    while current <= stop + 1e-9:
        values.append(round(current, 6))
        current += step
    return values


def metric_to_dict(metrics: Metrics) -> Dict[str, Optional[float]]:
    return {
        "score": metrics.score,
        "positive_count": metrics.positive_count,
        "negative_count": metrics.negative_count,
        "pos_diag_rate": metrics.pos_diag_rate,
        "pos_uncertain_rate": metrics.pos_uncertain_rate,
        "pos_invalid_rate": metrics.pos_invalid_rate,
        "pos_overall_acc": metrics.pos_overall_acc,
        "pos_diag_acc": metrics.pos_diag_acc,
        "neg_diag_rate": metrics.neg_diag_rate,
        "neg_uncertain_rate": metrics.neg_uncertain_rate,
        "neg_invalid_rate": metrics.neg_invalid_rate,
    }


def write_candidates_csv(rows: List[Dict], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "rank",
        "invalid_conf",
        "confident_conf",
        "margin",
        "score",
        "pos_diag_rate",
        "pos_uncertain_rate",
        "pos_invalid_rate",
        "pos_overall_acc",
        "pos_diag_acc",
        "neg_diag_rate",
        "neg_uncertain_rate",
        "neg_invalid_rate",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_metrics(name: str, th: Thresholds, m: Metrics) -> None:
    print(f"\n[{name}]")
    print(
        f"thresholds: invalid<{th.invalid_conf:.2f}, "
        f"confident>={th.confident_conf:.2f}, margin>={th.margin:.2f}"
    )
    print(
        f"positive: diag={m.pos_diag_rate:.4f}, uncertain={m.pos_uncertain_rate:.4f}, "
        f"invalid={m.pos_invalid_rate:.4f}, overall_acc={m.pos_overall_acc:.4f}, diag_acc={m.pos_diag_acc:.4f}"
    )
    if m.negative_count > 0:
        print(
            f"negative: diag={m.neg_diag_rate:.4f}, uncertain={m.neg_uncertain_rate:.4f}, "
            f"invalid={m.neg_invalid_rate:.4f}"
        )
    else:
        print("negative: N/A (negative-dir verilmedi)")
    print(f"score: {m.score:.4f}")


def main() -> None:
    args = parse_args()

    tflite_model_path = args.tflite_model.resolve()
    data_dir = args.data_dir.resolve()
    negative_dir = args.negative_dir.resolve() if args.negative_dir else None

    if not tflite_model_path.exists():
        raise FileNotFoundError(f"TFLite modeli bulunamadi: {tflite_model_path}")

    print(f"Pozitif veri yukleniyor: {data_dir}")
    pos_paths_all, pos_labels_all = collect_positive_dataset(data_dir)
    pos_paths, pos_labels = split_test_set(pos_paths_all, pos_labels_all, args.train_split, args.seed)

    if args.max_positive_samples > 0:
        pos_paths = pos_paths[: args.max_positive_samples]
        pos_labels = pos_labels[: args.max_positive_samples]

    neg_paths = None
    if negative_dir is not None:
        print(f"Negatif veri yukleniyor: {negative_dir}")
        neg_paths = collect_negative_dataset(negative_dir)
        if args.max_negative_samples > 0:
            neg_paths = neg_paths[: args.max_negative_samples]

    print(f"Pozitif test ornek sayisi: {len(pos_paths)}")
    if neg_paths is not None:
        print(f"Negatif ornek sayisi: {len(neg_paths)}")

    interpreter, input_detail, output_detail = build_interpreter(tflite_model_path, args.num_threads)

    print("Pozitif olasiliklar hesaplanıyor...")
    pos_probs = predict_probabilities(interpreter, input_detail, output_detail, pos_paths, args.image_size)
    pos_top1_idx, pos_top1_conf, pos_margin = to_features(pos_probs)

    neg_top1_conf = None
    neg_margin = None

    if neg_paths is not None and len(neg_paths) > 0:
        print("Negatif olasiliklar hesaplanıyor...")
        neg_probs = predict_probabilities(interpreter, input_detail, output_detail, neg_paths, args.image_size)
        _, neg_top1_conf, neg_margin = to_features(neg_probs)

    current_th = Thresholds(
        invalid_conf=CURRENT_INVALID_THRESHOLD,
        confident_conf=CURRENT_CONFIDENT_THRESHOLD,
        margin=CURRENT_MARGIN_THRESHOLD,
    )

    current_metrics = evaluate_thresholds(
        current_th,
        pos_top1_idx,
        pos_top1_conf,
        pos_margin,
        pos_labels,
        neg_top1_conf,
        neg_margin,
    )

    invalid_values = frange(args.invalid_min, args.invalid_max, args.invalid_step)
    confident_values = frange(args.confident_min, args.confident_max, args.confident_step)
    margin_values = frange(args.margin_min, args.margin_max, args.margin_step)

    rows: List[Dict] = []

    for invalid_conf in invalid_values:
        for confident_conf in confident_values:
            if invalid_conf >= confident_conf:
                continue
            for margin in margin_values:
                th = Thresholds(invalid_conf=invalid_conf, confident_conf=confident_conf, margin=margin)
                m = evaluate_thresholds(
                    th,
                    pos_top1_idx,
                    pos_top1_conf,
                    pos_margin,
                    pos_labels,
                    neg_top1_conf,
                    neg_margin,
                )

                rows.append(
                    {
                        "invalid_conf": invalid_conf,
                        "confident_conf": confident_conf,
                        "margin": margin,
                        **metric_to_dict(m),
                    }
                )

    if not rows:
        raise RuntimeError("Grid search adaylari olusturulamadi. Parametre araligini kontrol edin.")

    rows_sorted = sorted(rows, key=lambda x: x["score"], reverse=True)
    top_n = max(1, args.top_candidates)
    top_rows = rows_sorted[:top_n]

    best = top_rows[0]
    best_th = Thresholds(
        invalid_conf=float(best["invalid_conf"]),
        confident_conf=float(best["confident_conf"]),
        margin=float(best["margin"]),
    )
    best_metrics = evaluate_thresholds(
        best_th,
        pos_top1_idx,
        pos_top1_conf,
        pos_margin,
        pos_labels,
        neg_top1_conf,
        neg_margin,
    )

    report_json = args.report_json.resolve()
    report_csv = args.report_csv.resolve()
    report_json.parent.mkdir(parents=True, exist_ok=True)

    csv_rows: List[Dict] = []
    for rank, row in enumerate(top_rows, start=1):
        csv_rows.append(
            {
                "rank": rank,
                "invalid_conf": row["invalid_conf"],
                "confident_conf": row["confident_conf"],
                "margin": row["margin"],
                "score": row["score"],
                "pos_diag_rate": row["pos_diag_rate"],
                "pos_uncertain_rate": row["pos_uncertain_rate"],
                "pos_invalid_rate": row["pos_invalid_rate"],
                "pos_overall_acc": row["pos_overall_acc"],
                "pos_diag_acc": row["pos_diag_acc"],
                "neg_diag_rate": row["neg_diag_rate"],
                "neg_uncertain_rate": row["neg_uncertain_rate"],
                "neg_invalid_rate": row["neg_invalid_rate"],
            }
        )

    write_candidates_csv(csv_rows, report_csv)

    report = {
        "tflite_model": str(tflite_model_path),
        "positive_count": int(len(pos_paths)),
        "negative_count": int(len(neg_paths)) if neg_paths is not None else 0,
        "search_space": {
            "invalid_values": invalid_values,
            "confident_values": confident_values,
            "margin_values": margin_values,
        },
        "current_thresholds": {
            "invalid_conf": current_th.invalid_conf,
            "confident_conf": current_th.confident_conf,
            "margin": current_th.margin,
            "metrics": metric_to_dict(current_metrics),
        },
        "recommended_thresholds": {
            "invalid_conf": best_th.invalid_conf,
            "confident_conf": best_th.confident_conf,
            "margin": best_th.margin,
            "metrics": metric_to_dict(best_metrics),
        },
        "top_candidates_csv": str(report_csv),
    }

    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print_metrics("CURRENT", current_th, current_metrics)
    print_metrics("RECOMMENDED", best_th, best_metrics)

    print("\nKotlin snippet:")
    print("object DecisionThresholds {")
    print(f"    const val INVALID_IMAGE_CONFIDENCE_THRESHOLD = {best_th.invalid_conf:.2f}f")
    print(f"    const val CONFIDENT_DIAGNOSIS_THRESHOLD = {best_th.confident_conf:.2f}f")
    print(f"    const val MIN_MARGIN_THRESHOLD = {best_th.margin:.2f}f")
    print("}")

    print(f"\nJSON rapor: {report_json}")
    print(f"CSV adaylari: {report_csv}")


if __name__ == "__main__":
    main()
