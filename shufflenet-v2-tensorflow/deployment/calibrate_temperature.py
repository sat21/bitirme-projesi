#!/usr/bin/env python3
"""Calibrate temperature scaling for tomato TFLite inference.

This script searches a temperature value that reduces over-confident probabilities
while keeping diagnosis quality stable on in-distribution samples and reducing
false diagnosis on out-of-distribution negatives.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

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


@dataclass
class DistributionMetrics:
	nll: float
	ece: float
	top1_mean: float
	top1_p95: float


@dataclass
class StateMetrics:
	diagnosis_rate: float
	uncertain_rate: float
	invalid_rate: float
	diag_acc: float
	overall_acc: float


@dataclass
class NegativeStateMetrics:
	diagnosis_rate: float
	uncertain_rate: float
	invalid_rate: float
	top1_mean: float
	top1_p95: float


def parse_args() -> argparse.Namespace:
	project_root = Path(__file__).resolve().parents[1]
	default_tflite = project_root / "artifacts" / "tflite" / "checkpoints_tomato_1_5x_baseline_best_model_int8.tflite"
	default_data = project_root.parent / "tomato"
	default_report_json = project_root / "artifacts" / "tflite" / "temperature_calibration_report.json"
	default_report_csv = project_root / "artifacts" / "tflite" / "temperature_calibration_candidates.csv"

	parser = argparse.ArgumentParser(description="Calibrate temperature scaling for mobile probabilities.")
	parser.add_argument("--tflite-model", type=Path, default=default_tflite)
	parser.add_argument("--data-dir", type=Path, default=default_data)
	parser.add_argument("--negative-dir", type=Path, default=None)
	parser.add_argument("--image-size", type=int, default=224)
	parser.add_argument("--calibration-split", type=float, default=0.5)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--max-positive-samples", type=int, default=0)
	parser.add_argument("--max-negative-samples", type=int, default=0)
	parser.add_argument("--num-threads", type=int, default=4)
	parser.add_argument("--temperature-min", type=float, default=0.80)
	parser.add_argument("--temperature-max", type=float, default=3.20)
	parser.add_argument("--temperature-step", type=float, default=0.10)
	parser.add_argument("--current-temperature", type=float, default=2.40)
	parser.add_argument("--invalid-conf", type=float, default=0.70)
	parser.add_argument("--confident-conf", type=float, default=0.90)
	parser.add_argument("--margin", type=float, default=0.10)
	parser.add_argument("--min-top3-mass", type=float, default=0.88)
	parser.add_argument("--max-normalized-entropy", type=float, default=0.50)
	parser.add_argument("--ece-bins", type=int, default=15)
	parser.add_argument("--top-candidates", type=int, default=20)
	parser.add_argument("--min-pos-diag-rate", type=float, default=0.94)
	parser.add_argument("--min-pos-diag-acc", type=float, default=0.98)
	parser.add_argument("--max-pos-invalid-rate", type=float, default=0.03)
	parser.add_argument("--report-json", type=Path, default=default_report_json)
	parser.add_argument("--report-csv", type=Path, default=default_report_csv)
	return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
	if not args.tflite_model.exists():
		raise FileNotFoundError(f"Model not found: {args.tflite_model}")
	if not args.data_dir.exists():
		raise FileNotFoundError(f"Data directory not found: {args.data_dir}")
	if args.negative_dir is not None and not args.negative_dir.exists():
		raise FileNotFoundError(f"Negative directory not found: {args.negative_dir}")
	if not 0.1 <= args.calibration_split <= 0.9:
		raise ValueError("calibration-split must be in [0.1, 0.9].")
	if args.temperature_step <= 0:
		raise ValueError("temperature-step must be > 0.")
	if args.temperature_min <= 0 or args.temperature_max <= 0:
		raise ValueError("temperature values must be > 0.")
	if args.temperature_min > args.temperature_max:
		raise ValueError("temperature-min cannot be larger than temperature-max.")


def collect_positive_dataset(data_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
	patterns = ("*.jpg", "*.JPG", "*.jpeg", "*.png")
	image_paths: List[str] = []
	labels: List[int] = []

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
		raise ValueError(f"No positive images found under: {data_dir}")

	return np.array(image_paths), np.array(labels, dtype=np.int32)


def collect_negative_dataset(negative_dir: Path) -> np.ndarray:
	patterns = ("*.jpg", "*.JPG", "*.jpeg", "*.png")
	files: List[Path] = []
	for pattern in patterns:
		files.extend(sorted(negative_dir.rglob(pattern)))
	if not files:
		raise ValueError(f"No negative images found under: {negative_dir}")
	return np.array([str(path) for path in files])


def stratified_split(
	image_paths: np.ndarray,
	labels: np.ndarray,
	calibration_split: float,
	seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	rng = np.random.default_rng(seed)
	calibration_indices: List[int] = []
	evaluation_indices: List[int] = []

	for class_idx in np.unique(labels):
		class_indices = np.where(labels == class_idx)[0]
		shuffled = rng.permutation(class_indices)
		split_idx = max(1, int(len(shuffled) * calibration_split))
		split_idx = min(split_idx, len(shuffled) - 1) if len(shuffled) > 1 else len(shuffled)

		calibration_indices.extend(shuffled[:split_idx])
		evaluation_indices.extend(shuffled[split_idx:])

	calibration_indices = np.array(calibration_indices, dtype=np.int32)
	evaluation_indices = np.array(evaluation_indices, dtype=np.int32)

	if len(evaluation_indices) == 0:
		raise ValueError("Evaluation split is empty. Increase dataset size or adjust calibration-split.")

	return (
		image_paths[calibration_indices],
		labels[calibration_indices],
		image_paths[evaluation_indices],
		labels[evaluation_indices],
	)


def maybe_subsample(
	image_paths: np.ndarray,
	labels: Optional[np.ndarray],
	max_samples: int,
	seed: int,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
	if max_samples <= 0 or len(image_paths) <= max_samples:
		return image_paths, labels
	rng = np.random.default_rng(seed)
	indices = rng.choice(len(image_paths), size=max_samples, replace=False)
	indices.sort()
	if labels is None:
		return image_paths[indices], None
	return image_paths[indices], labels[indices]


def preprocess_image(image_path: Path, image_size: int) -> np.ndarray:
	image = Image.open(image_path).convert("RGB")
	image = image.resize((image_size, image_size), Image.BILINEAR)
	arr = np.asarray(image, dtype=np.float32) / 255.0
	arr = (arr - 0.5) / 0.5
	return np.expand_dims(arr, axis=0).astype(np.float32)


def build_interpreter(tflite_path: Path, num_threads: int):
	interpreter = tf.lite.Interpreter(model_path=str(tflite_path), num_threads=num_threads)
	interpreter.allocate_tensors()
	input_detail = interpreter.get_input_details()[0]
	output_detail = interpreter.get_output_details()[0]
	return interpreter, input_detail, output_detail


def quantize_input(input_tensor: np.ndarray, input_detail: Dict) -> np.ndarray:
	dtype = input_detail["dtype"]
	if dtype == np.float32:
		return input_tensor.astype(np.float32)

	scale, zero_point = input_detail["quantization"]
	if scale == 0:
		raise ValueError("Input quantization scale cannot be 0.")

	quantized = np.round(input_tensor / scale + zero_point)

	if dtype == np.int8:
		quantized = np.clip(quantized, -128, 127).astype(np.int8)
	elif dtype == np.uint8:
		quantized = np.clip(quantized, 0, 255).astype(np.uint8)
	else:
		raise ValueError(f"Unsupported input dtype: {dtype}")

	return quantized


def dequantize_output(output_tensor: np.ndarray, output_detail: Dict) -> np.ndarray:
	dtype = output_detail["dtype"]
	if dtype == np.float32:
		return output_tensor.astype(np.float32)

	scale, zero_point = output_detail["quantization"]
	if scale == 0:
		raise ValueError("Output quantization scale cannot be 0.")

	return (output_tensor.astype(np.float32) - zero_point) * scale


def infer_raw_scores(
	interpreter,
	input_detail: Dict,
	output_detail: Dict,
	image_paths: np.ndarray,
	image_size: int,
) -> np.ndarray:
	scores = np.zeros((len(image_paths), len(CLASS_NAMES)), dtype=np.float32)

	input_index = input_detail["index"]
	output_index = output_detail["index"]

	for idx, image_path in enumerate(image_paths):
		input_tensor = preprocess_image(Path(image_path), image_size)
		model_input = quantize_input(input_tensor, input_detail)

		interpreter.set_tensor(input_index, model_input)
		interpreter.invoke()

		raw_output = interpreter.get_tensor(output_index)
		raw_scores = dequantize_output(raw_output[0], output_detail)
		scores[idx] = raw_scores

		if (idx + 1) % 500 == 0 or (idx + 1) == len(image_paths):
			print(f"[inference] {idx + 1}/{len(image_paths)}")

	return scores


def softmax_batch(logits: np.ndarray, temperature: float) -> np.ndarray:
	safe_temperature = max(temperature, 0.05)
	scaled = logits / safe_temperature
	shifted = scaled - np.max(scaled, axis=1, keepdims=True)
	exp_scores = np.exp(shifted)
	sums = np.sum(exp_scores, axis=1, keepdims=True)
	sums = np.clip(sums, 1e-12, None)
	return exp_scores / sums


def normalized_entropy_batch(probabilities: np.ndarray) -> np.ndarray:
	probs = np.clip(probabilities, 1e-12, 1.0)
	entropy = -np.sum(probs * np.log(probs), axis=1)
	max_entropy = np.log(probabilities.shape[1])
	return np.clip(entropy / max(max_entropy, 1e-12), 0.0, 1.0)


def topk_metrics(probabilities: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	order = np.argsort(probabilities, axis=1)[:, ::-1]
	top1_idx = order[:, 0]
	top1_conf = probabilities[np.arange(len(probabilities)), top1_idx]
	top2_conf = probabilities[np.arange(len(probabilities)), order[:, 1]]
	margin = top1_conf - top2_conf
	return top1_idx, top1_conf, margin


def expected_calibration_error(probabilities: np.ndarray, labels: np.ndarray, num_bins: int) -> float:
	confidences = np.max(probabilities, axis=1)
	predictions = np.argmax(probabilities, axis=1)
	correctness = (predictions == labels).astype(np.float32)

	edges = np.linspace(0.0, 1.0, num_bins + 1)
	ece = 0.0
	total = len(confidences)

	for i in range(num_bins):
		lower, upper = edges[i], edges[i + 1]
		if i == num_bins - 1:
			mask = (confidences >= lower) & (confidences <= upper)
		else:
			mask = (confidences >= lower) & (confidences < upper)

		if not np.any(mask):
			continue

		bin_acc = float(np.mean(correctness[mask]))
		bin_conf = float(np.mean(confidences[mask]))
		bin_weight = float(np.sum(mask) / total)
		ece += abs(bin_acc - bin_conf) * bin_weight

	return float(ece)


def negative_log_likelihood(probabilities: np.ndarray, labels: np.ndarray) -> float:
	probs_true = probabilities[np.arange(len(labels)), labels]
	probs_true = np.clip(probs_true, 1e-12, 1.0)
	return float(-np.mean(np.log(probs_true)))


def classify_states(
	top1_conf: np.ndarray,
	margin: np.ndarray,
	top3_mass: np.ndarray,
	normalized_entropy: np.ndarray,
	invalid_conf: float,
	confident_conf: float,
	margin_threshold: float,
	min_top3_mass: float,
	max_normalized_entropy: float,
) -> np.ndarray:
	states = np.full(len(top1_conf), "DIAGNOSIS", dtype=object)
	invalid_mask = top1_conf < invalid_conf
	uncertain_mask = (~invalid_mask) & (
		(top1_conf < confident_conf)
		| (margin < margin_threshold)
		| (top3_mass < min_top3_mass)
		| (normalized_entropy > max_normalized_entropy)
	)

	states[invalid_mask] = "INVALID_IMAGE"
	states[uncertain_mask] = "UNCERTAIN"
	return states


def evaluate_positive_states(probabilities: np.ndarray, labels: np.ndarray, args: argparse.Namespace) -> StateMetrics:
	top1_idx, top1_conf, margin = topk_metrics(probabilities)
	top3_mass = np.sum(np.sort(probabilities, axis=1)[:, -3:], axis=1)
	norm_entropy = normalized_entropy_batch(probabilities)

	states = classify_states(
		top1_conf=top1_conf,
		margin=margin,
		top3_mass=top3_mass,
		normalized_entropy=norm_entropy,
		invalid_conf=args.invalid_conf,
		confident_conf=args.confident_conf,
		margin_threshold=args.margin,
		min_top3_mass=args.min_top3_mass,
		max_normalized_entropy=args.max_normalized_entropy,
	)

	diagnosis_mask = states == "DIAGNOSIS"
	uncertain_mask = states == "UNCERTAIN"
	invalid_mask = states == "INVALID_IMAGE"

	correct = top1_idx == labels
	diag_acc = float(np.mean(correct[diagnosis_mask])) if np.any(diagnosis_mask) else 0.0
	overall_acc = float(np.mean(correct))

	return StateMetrics(
		diagnosis_rate=float(np.mean(diagnosis_mask)),
		uncertain_rate=float(np.mean(uncertain_mask)),
		invalid_rate=float(np.mean(invalid_mask)),
		diag_acc=diag_acc,
		overall_acc=overall_acc,
	)


def evaluate_negative_states(probabilities: np.ndarray, args: argparse.Namespace) -> NegativeStateMetrics:
	_, top1_conf, margin = topk_metrics(probabilities)
	top3_mass = np.sum(np.sort(probabilities, axis=1)[:, -3:], axis=1)
	norm_entropy = normalized_entropy_batch(probabilities)

	states = classify_states(
		top1_conf=top1_conf,
		margin=margin,
		top3_mass=top3_mass,
		normalized_entropy=norm_entropy,
		invalid_conf=args.invalid_conf,
		confident_conf=args.confident_conf,
		margin_threshold=args.margin,
		min_top3_mass=args.min_top3_mass,
		max_normalized_entropy=args.max_normalized_entropy,
	)

	return NegativeStateMetrics(
		diagnosis_rate=float(np.mean(states == "DIAGNOSIS")),
		uncertain_rate=float(np.mean(states == "UNCERTAIN")),
		invalid_rate=float(np.mean(states == "INVALID_IMAGE")),
		top1_mean=float(np.mean(top1_conf)),
		top1_p95=float(np.percentile(top1_conf, 95)),
	)


def evaluate_distribution(probabilities: np.ndarray, labels: np.ndarray, num_bins: int) -> DistributionMetrics:
	_, top1_conf, _ = topk_metrics(probabilities)
	return DistributionMetrics(
		nll=negative_log_likelihood(probabilities, labels),
		ece=expected_calibration_error(probabilities, labels, num_bins=num_bins),
		top1_mean=float(np.mean(top1_conf)),
		top1_p95=float(np.percentile(top1_conf, 95)),
	)


def objective_value(
	distribution: DistributionMetrics,
	positive_states: StateMetrics,
	negative_states: Optional[NegativeStateMetrics],
) -> float:
	score = 0.0
	score += 0.50 * distribution.nll
	score += 0.45 * distribution.ece
	score += 1.30 * positive_states.invalid_rate
	score += 0.80 * (1.0 - positive_states.diag_acc)

	if negative_states is not None:
		score += 1.50 * negative_states.diagnosis_rate
		score += 0.30 * (1.0 - negative_states.invalid_rate)

	return float(score)


def frange(min_value: float, max_value: float, step: float) -> Iterable[float]:
	values: List[float] = []
	v = min_value
	while v <= max_value + 1e-9:
		values.append(round(v, 6))
		v += step
	return values


def write_candidates_csv(rows: List[Dict[str, float]], csv_path: Path) -> None:
	csv_path.parent.mkdir(parents=True, exist_ok=True)
	if not rows:
		return

	headers = list(rows[0].keys())
	with csv_path.open("w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=headers)
		writer.writeheader()
		writer.writerows(rows)


def run() -> None:
	args = parse_args()
	validate_args(args)

	print("[info] loading datasets...")
	positive_paths, positive_labels = collect_positive_dataset(args.data_dir)

	positive_paths, positive_labels = maybe_subsample(
		positive_paths,
		positive_labels,
		max_samples=args.max_positive_samples,
		seed=args.seed,
	)

	cal_paths, cal_labels, eval_paths, eval_labels = stratified_split(
		positive_paths,
		positive_labels,
		calibration_split=args.calibration_split,
		seed=args.seed,
	)

	negative_paths: Optional[np.ndarray] = None
	if args.negative_dir is not None:
		negative_paths = collect_negative_dataset(args.negative_dir)
		negative_paths, _ = maybe_subsample(
			negative_paths,
			labels=None,
			max_samples=args.max_negative_samples,
			seed=args.seed,
		)

	print(f"[info] calibration positives: {len(cal_paths)}")
	print(f"[info] evaluation positives: {len(eval_paths)}")
	if negative_paths is not None:
		print(f"[info] evaluation negatives: {len(negative_paths)}")

	print("[info] running TFLite inference once per image...")
	interpreter, input_detail, output_detail = build_interpreter(args.tflite_model, args.num_threads)

	cal_scores = infer_raw_scores(interpreter, input_detail, output_detail, cal_paths, args.image_size)
	eval_scores = infer_raw_scores(interpreter, input_detail, output_detail, eval_paths, args.image_size)
	neg_scores = None
	if negative_paths is not None:
		neg_scores = infer_raw_scores(interpreter, input_detail, output_detail, negative_paths, args.image_size)

	temperatures = list(frange(args.temperature_min, args.temperature_max, args.temperature_step))
	if args.current_temperature not in temperatures:
		temperatures.append(round(args.current_temperature, 6))
	if 1.0 not in temperatures:
		temperatures.append(1.0)
	temperatures = sorted(set(temperatures))

	rows: List[Dict[str, float]] = []
	best_temperature = None
	best_score = float("inf")
	best_bundle = None
	best_feasible_temperature = None
	best_feasible_score = float("inf")
	best_feasible_bundle = None

	for temperature in temperatures:
		cal_probs = softmax_batch(cal_scores, temperature)
		eval_probs = softmax_batch(eval_scores, temperature)

		dist_metrics = evaluate_distribution(eval_probs, eval_labels, num_bins=args.ece_bins)
		cal_nll = negative_log_likelihood(cal_probs, cal_labels)
		dist_metrics = DistributionMetrics(
			nll=cal_nll,
			ece=dist_metrics.ece,
			top1_mean=dist_metrics.top1_mean,
			top1_p95=dist_metrics.top1_p95,
		)

		pos_states = evaluate_positive_states(eval_probs, eval_labels, args)

		neg_states = None
		if neg_scores is not None:
			neg_probs = softmax_batch(neg_scores, temperature)
			neg_states = evaluate_negative_states(neg_probs, args)

		score = objective_value(dist_metrics, pos_states, neg_states)

		row: Dict[str, float] = {
			"temperature": float(temperature),
			"objective": score,
			"cal_nll": dist_metrics.nll,
			"eval_ece": dist_metrics.ece,
			"eval_pos_top1_mean": dist_metrics.top1_mean,
			"eval_pos_top1_p95": dist_metrics.top1_p95,
			"eval_pos_diag_rate": pos_states.diagnosis_rate,
			"eval_pos_uncertain_rate": pos_states.uncertain_rate,
			"eval_pos_invalid_rate": pos_states.invalid_rate,
			"eval_pos_diag_acc": pos_states.diag_acc,
			"eval_pos_overall_acc": pos_states.overall_acc,
		}

		if neg_states is not None:
			row.update(
				{
					"eval_neg_diag_rate": neg_states.diagnosis_rate,
					"eval_neg_uncertain_rate": neg_states.uncertain_rate,
					"eval_neg_invalid_rate": neg_states.invalid_rate,
					"eval_neg_top1_mean": neg_states.top1_mean,
					"eval_neg_top1_p95": neg_states.top1_p95,
				}
			)

		rows.append(row)

		if score < best_score:
			best_score = score
			best_temperature = temperature
			best_bundle = (dist_metrics, pos_states, neg_states)

		is_feasible = (
			pos_states.diagnosis_rate >= args.min_pos_diag_rate
			and pos_states.diag_acc >= args.min_pos_diag_acc
			and pos_states.invalid_rate <= args.max_pos_invalid_rate
		)
		if is_feasible and score < best_feasible_score:
			best_feasible_score = score
			best_feasible_temperature = temperature
			best_feasible_bundle = (dist_metrics, pos_states, neg_states)

	rows = sorted(rows, key=lambda item: item["objective"])
	top_rows = rows[: max(1, args.top_candidates)]

	selected_temperature = best_feasible_temperature if best_feasible_temperature is not None else best_temperature
	selected_score = best_feasible_score if best_feasible_temperature is not None else best_score
	selected_bundle = best_feasible_bundle if best_feasible_temperature is not None else best_bundle

	write_candidates_csv(top_rows, args.report_csv)

	baseline_row = next((r for r in rows if abs(r["temperature"] - 1.0) < 1e-9), None)
	current_row = next((r for r in rows if abs(r["temperature"] - args.current_temperature) < 1e-9), None)
	selected_row = next((r for r in rows if abs(r["temperature"] - selected_temperature) < 1e-9), None)

	report = {
		"config": {
			"tflite_model": str(args.tflite_model),
			"data_dir": str(args.data_dir),
			"negative_dir": str(args.negative_dir) if args.negative_dir else None,
			"image_size": args.image_size,
			"calibration_split": args.calibration_split,
			"seed": args.seed,
			"temperature_min": args.temperature_min,
			"temperature_max": args.temperature_max,
			"temperature_step": args.temperature_step,
			"current_temperature": args.current_temperature,
			"selection_guardrails": {
				"min_pos_diag_rate": args.min_pos_diag_rate,
				"min_pos_diag_acc": args.min_pos_diag_acc,
				"max_pos_invalid_rate": args.max_pos_invalid_rate,
			},
			"decision_thresholds": {
				"invalid_conf": args.invalid_conf,
				"confident_conf": args.confident_conf,
				"margin": args.margin,
				"min_top3_mass": args.min_top3_mass,
				"max_normalized_entropy": args.max_normalized_entropy,
			},
		},
		"dataset": {
			"calibration_positive_count": len(cal_paths),
			"evaluation_positive_count": len(eval_paths),
			"evaluation_negative_count": 0 if negative_paths is None else len(negative_paths),
		},
		"selected_from_feasible_set": best_feasible_temperature is not None,
		"baseline_temperature_1_0": baseline_row,
		"current_temperature": current_row,
		"recommended_temperature": selected_row,
		"top_candidates_csv": str(args.report_csv),
	}

	args.report_json.parent.mkdir(parents=True, exist_ok=True)
	args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

	print("\n[RESULT]")
	print(f"recommended_temperature={selected_temperature:.2f}")
	print(f"objective={selected_score:.6f}")
	if best_feasible_temperature is None:
		print("selection_mode=global_best_no_feasible_candidate")
	else:
		print("selection_mode=feasible_guardrail_best")

	if selected_bundle is not None:
		dist_metrics, pos_states, neg_states = selected_bundle
		print(f"cal_nll={dist_metrics.nll:.6f}")
		print(f"eval_ece={dist_metrics.ece:.6f}")
		print(f"eval_pos_diag_rate={pos_states.diagnosis_rate:.4f}")
		print(f"eval_pos_invalid_rate={pos_states.invalid_rate:.4f}")
		print(f"eval_pos_diag_acc={pos_states.diag_acc:.4f}")
		if neg_states is not None:
			print(f"eval_neg_diag_rate={neg_states.diagnosis_rate:.4f}")
			print(f"eval_neg_invalid_rate={neg_states.invalid_rate:.4f}")

	print("\n[KOTLIN SNIPPET]")
	print("object ModelCalibration {")
	print("    // Recommended by calibrate_temperature.py")
	print(f"    const val TEMPERATURE_SCALING_FACTOR = {selected_temperature:.2f}f")
	print("}")
	print(f"\nreport_json={args.report_json}")
	print(f"report_csv={args.report_csv}")


if __name__ == "__main__":
	run()
