#!/usr/bin/env python3
"""Train a two-stage binary classifier: healthy vs disease.

This script is the first stage of a production pipeline:
1) Validate leaf image + classify healthy/disease.
2) If disease, run disease-subclass model.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
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

HEALTHY_CLASS_NAME = "Tomato___healthy"


@dataclass
class SplitDataset:
	train_paths: np.ndarray
	train_labels: np.ndarray
	val_paths: np.ndarray
	val_labels: np.ndarray
	test_paths: np.ndarray
	test_labels: np.ndarray


def parse_args() -> argparse.Namespace:
	project_root = Path(__file__).resolve().parent
	default_data_dir = project_root.parent / "tomato"
	default_output_dir = project_root / "artifacts" / "two_stage"

	parser = argparse.ArgumentParser(description="Train healthy-vs-disease stage for two-stage diagnosis.")
	parser.add_argument("--data-dir", type=Path, default=default_data_dir)
	parser.add_argument("--output-dir", type=Path, default=default_output_dir)
	parser.add_argument("--image-size", type=int, default=224)
	parser.add_argument("--batch-size", type=int, default=32)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--train-ratio", type=float, default=0.70)
	parser.add_argument("--val-ratio", type=float, default=0.15)
	parser.add_argument("--epochs-head", type=int, default=6)
	parser.add_argument("--epochs-finetune", type=int, default=10)
	parser.add_argument("--fine-tune-layers", type=int, default=50)
	parser.add_argument("--head-lr", type=float, default=1e-3)
	parser.add_argument("--finetune-lr", type=float, default=1e-4)
	parser.add_argument("--dropout", type=float, default=0.30)
	parser.add_argument("--label-smoothing", type=float, default=0.05)
	parser.add_argument("--max-samples", type=int, default=0)
	parser.add_argument("--export-tflite", action="store_true")
	parser.add_argument("--dry-run", action="store_true")
	return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
	if not args.data_dir.exists():
		raise FileNotFoundError(f"Data directory not found: {args.data_dir}")
	if not 0.0 < args.train_ratio < 1.0:
		raise ValueError("train-ratio must be in (0, 1).")
	if not 0.0 < args.val_ratio < 1.0:
		raise ValueError("val-ratio must be in (0, 1).")
	if args.train_ratio + args.val_ratio >= 1.0:
		raise ValueError("train-ratio + val-ratio must be < 1.0 to leave test split.")


def collect_binary_dataset(data_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
	patterns = ("*.jpg", "*.JPG", "*.jpeg", "*.png")
	paths: List[str] = []
	labels: List[int] = []

	for class_name in CLASS_NAMES:
		class_dir = data_dir / class_name
		if not class_dir.exists():
			continue

		files: List[Path] = []
		for pattern in patterns:
			files.extend(sorted(class_dir.glob(pattern)))

		if not files:
			continue

		binary_label = 0 if class_name == HEALTHY_CLASS_NAME else 1
		paths.extend(str(path) for path in files)
		labels.extend([binary_label] * len(files))

	if not paths:
		raise ValueError(f"No images found under: {data_dir}")

	return np.array(paths), np.array(labels, dtype=np.int32)


def maybe_subsample(paths: np.ndarray, labels: np.ndarray, max_samples: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
	if max_samples <= 0 or len(paths) <= max_samples:
		return paths, labels
	rng = np.random.default_rng(seed)
	indices = rng.choice(len(paths), size=max_samples, replace=False)
	indices.sort()
	return paths[indices], labels[indices]


def stratified_split(paths: np.ndarray, labels: np.ndarray, train_ratio: float, val_ratio: float, seed: int) -> SplitDataset:
	rng = np.random.default_rng(seed)
	train_idx: List[int] = []
	val_idx: List[int] = []
	test_idx: List[int] = []

	for label in (0, 1):
		indices = np.where(labels == label)[0]
		shuffled = rng.permutation(indices)

		n = len(shuffled)
		n_train = max(1, int(n * train_ratio))
		n_val = max(1, int(n * val_ratio))
		n_test = n - n_train - n_val
		if n_test <= 0:
			n_test = 1
			if n_train > n_val:
				n_train -= 1
			else:
				n_val -= 1

		train_idx.extend(shuffled[:n_train])
		val_idx.extend(shuffled[n_train:n_train + n_val])
		test_idx.extend(shuffled[n_train + n_val:n_train + n_val + n_test])

	train_idx = np.array(train_idx, dtype=np.int32)
	val_idx = np.array(val_idx, dtype=np.int32)
	test_idx = np.array(test_idx, dtype=np.int32)

	return SplitDataset(
		train_paths=paths[train_idx],
		train_labels=labels[train_idx],
		val_paths=paths[val_idx],
		val_labels=labels[val_idx],
		test_paths=paths[test_idx],
		test_labels=labels[test_idx],
	)


def decode_resize(path: tf.Tensor, label: tf.Tensor, image_size: int) -> Tuple[tf.Tensor, tf.Tensor]:
	image = tf.io.read_file(path)
	image = tf.image.decode_image(image, channels=3, expand_animations=False)
	image = tf.image.resize(image, [image_size, image_size], method=tf.image.ResizeMethod.BILINEAR)
	image = tf.cast(image, tf.float32)
	return image, tf.cast(label, tf.float32)


def make_dataset(paths: np.ndarray, labels: np.ndarray, image_size: int, batch_size: int, training: bool, seed: int) -> tf.data.Dataset:
	ds = tf.data.Dataset.from_tensor_slices((paths, labels))
	if training:
		ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
	ds = ds.map(lambda p, y: decode_resize(p, y, image_size), num_parallel_calls=tf.data.AUTOTUNE)
	ds = ds.batch(batch_size)
	ds = ds.prefetch(tf.data.AUTOTUNE)
	return ds


def build_model(image_size: int, dropout: float) -> Tuple[tf.keras.Model, tf.keras.Model]:
	inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="input_image")
	augmentation = tf.keras.Sequential(
		[
			tf.keras.layers.RandomFlip("horizontal"),
			tf.keras.layers.RandomRotation(0.08),
			tf.keras.layers.RandomZoom(0.12),
			tf.keras.layers.RandomContrast(0.10),
		],
		name="augmentation",
	)

	x = augmentation(inputs)
	x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

	base_model = tf.keras.applications.MobileNetV2(
		include_top=False,
		weights="imagenet",
		input_shape=(image_size, image_size, 3),
	)
	base_model.trainable = False

	x = base_model(x, training=False)
	x = tf.keras.layers.GlobalAveragePooling2D()(x)
	x = tf.keras.layers.Dropout(dropout)(x)
	outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="healthy_vs_disease")(x)

	model = tf.keras.Model(inputs=inputs, outputs=outputs, name="tomato_two_stage_binary")
	return model, base_model


def class_weights(labels: np.ndarray) -> Dict[int, float]:
	count_healthy = int(np.sum(labels == 0))
	count_disease = int(np.sum(labels == 1))
	total = count_healthy + count_disease

	w_healthy = total / max(2 * count_healthy, 1)
	w_disease = total / max(2 * count_disease, 1)
	return {0: float(w_healthy), 1: float(w_disease)}


def compile_model(model: tf.keras.Model, learning_rate: float, label_smoothing: float) -> None:
	model.compile(
		optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
		loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing),
		metrics=[
			tf.keras.metrics.BinaryAccuracy(name="accuracy"),
			tf.keras.metrics.AUC(name="auc"),
			tf.keras.metrics.Precision(name="precision"),
			tf.keras.metrics.Recall(name="recall"),
		],
	)


def export_tflite(model: tf.keras.Model, output_path: Path) -> None:
	converter = tf.lite.TFLiteConverter.from_keras_model(model)
	tflite_model = converter.convert()
	output_path.write_bytes(tflite_model)


def evaluate_and_save_report(
	model: tf.keras.Model,
	test_ds: tf.data.Dataset,
	output_dir: Path,
	split: SplitDataset,
	args: argparse.Namespace,
) -> None:
	results = model.evaluate(test_ds, verbose=0)
	metrics = dict(zip(model.metrics_names, [float(v) for v in results]))

	report = {
		"model": "tomato_two_stage_binary",
		"healthy_label": 0,
		"disease_label": 1,
		"dataset": {
			"train_count": len(split.train_paths),
			"val_count": len(split.val_paths),
			"test_count": len(split.test_paths),
			"train_healthy": int(np.sum(split.train_labels == 0)),
			"train_disease": int(np.sum(split.train_labels == 1)),
		},
		"training": {
			"epochs_head": args.epochs_head,
			"epochs_finetune": args.epochs_finetune,
			"head_lr": args.head_lr,
			"finetune_lr": args.finetune_lr,
			"label_smoothing": args.label_smoothing,
			"dropout": args.dropout,
			"batch_size": args.batch_size,
			"image_size": args.image_size,
		},
		"test_metrics": metrics,
	}

	report_path = output_dir / "two_stage_training_report.json"
	report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
	print(f"[saved] report: {report_path}")


def run() -> None:
	args = parse_args()
	validate_args(args)
	tf.keras.utils.set_random_seed(args.seed)

	args.output_dir.mkdir(parents=True, exist_ok=True)

	print("[info] collecting dataset...")
	paths, labels = collect_binary_dataset(args.data_dir)
	paths, labels = maybe_subsample(paths, labels, args.max_samples, args.seed)
	split = stratified_split(paths, labels, args.train_ratio, args.val_ratio, args.seed)

	print(f"[info] train={len(split.train_paths)} val={len(split.val_paths)} test={len(split.test_paths)}")
	print(f"[info] train healthy={np.sum(split.train_labels == 0)} disease={np.sum(split.train_labels == 1)}")

	if args.dry_run:
		print("[dry-run] dataset and split look valid. Training skipped.")
		return

	train_ds = make_dataset(split.train_paths, split.train_labels, args.image_size, args.batch_size, training=True, seed=args.seed)
	val_ds = make_dataset(split.val_paths, split.val_labels, args.image_size, args.batch_size, training=False, seed=args.seed)
	test_ds = make_dataset(split.test_paths, split.test_labels, args.image_size, args.batch_size, training=False, seed=args.seed)

	model, base_model = build_model(args.image_size, args.dropout)

	callbacks = [
		tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=4, restore_best_weights=True),
		tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
		tf.keras.callbacks.ModelCheckpoint(
			filepath=str(args.output_dir / "two_stage_best.keras"),
			monitor="val_auc",
			mode="max",
			save_best_only=True,
		),
	]

	weights = class_weights(split.train_labels)
	print(f"[info] class weights: {weights}")

	print("[phase-1] train classification head")
	compile_model(model, learning_rate=args.head_lr, label_smoothing=args.label_smoothing)
	model.fit(
		train_ds,
		validation_data=val_ds,
		epochs=args.epochs_head,
		callbacks=callbacks,
		class_weight=weights,
		verbose=1,
	)

	print("[phase-2] fine-tune last layers")
	base_model.trainable = True
	if args.fine_tune_layers > 0:
		freeze_until = max(0, len(base_model.layers) - args.fine_tune_layers)
		for layer in base_model.layers[:freeze_until]:
			layer.trainable = False

	compile_model(model, learning_rate=args.finetune_lr, label_smoothing=args.label_smoothing)
	model.fit(
		train_ds,
		validation_data=val_ds,
		epochs=args.epochs_head + args.epochs_finetune,
		initial_epoch=args.epochs_head,
		callbacks=callbacks,
		class_weight=weights,
		verbose=1,
	)

	final_model_path = args.output_dir / "two_stage_final.keras"
	model.save(final_model_path)
	print(f"[saved] keras model: {final_model_path}")

	if args.export_tflite:
		tflite_path = args.output_dir / "two_stage_final.tflite"
		export_tflite(model, tflite_path)
		print(f"[saved] tflite model: {tflite_path}")

	evaluate_and_save_report(model, test_ds, args.output_dir, split, args)


if __name__ == "__main__":
	run()
