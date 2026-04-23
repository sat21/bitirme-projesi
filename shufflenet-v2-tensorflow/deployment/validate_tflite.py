#!/usr/bin/env python3
"""Compare Keras and TFLite model outputs on the tomato dataset."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import time
from typing import List, Tuple

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


class ChannelShuffle(tf.keras.layers.Layer):
    def __init__(self, groups: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.groups = groups

    def call(self, x):
        shape = tf.shape(x)
        batch_size = shape[0]
        height = shape[1]
        width = shape[2]
        channels = shape[3]
        channels_per_group = channels // self.groups
        x = tf.reshape(x, [batch_size, height, width, self.groups, channels_per_group])
        x = tf.transpose(x, [0, 1, 2, 4, 3])
        x = tf.reshape(x, [batch_size, height, width, channels])
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"groups": self.groups})
        return config


class ChannelSplit(tf.keras.layers.Layer):
    def __init__(self, split_idx: int, **kwargs):
        super().__init__(**kwargs)
        self.split_idx = split_idx

    def call(self, x):
        return tf.split(x, num_or_size_splits=2, axis=-1)[self.split_idx]

    def get_config(self):
        config = super().get_config()
        config.update({"split_idx": self.split_idx})
        return config


def channel_shuffle(x, groups: int):
    shape = tf.shape(x)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]
    channels = shape[3]
    channels_per_group = channels // groups
    x = tf.reshape(x, [batch_size, height, width, groups, channels_per_group])
    x = tf.transpose(x, [0, 1, 2, 4, 3])
    x = tf.reshape(x, [batch_size, height, width, channels])
    return x


def get_custom_objects() -> dict:
    return {
        "ChannelShuffle": ChannelShuffle,
        "ChannelSplit": ChannelSplit,
        "ChannelShuffleLayer": ChannelShuffle,
        "ChannelSplitLayer": ChannelSplit,
        "channel_shuffle": channel_shuffle,
    }


def get_default_data_dir(project_root: Path) -> Path:
    candidate = project_root.parent / "tomato"
    if candidate.exists():
        return candidate
    return Path("/mnt/021630F41630E9F5/PROJECTS/torch/tomato")


def preprocess_image(image_path: Path, image_size: int) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    img = img.resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return arr


def collect_dataset(data_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    image_paths: List[str] = []
    labels: List[int] = []
    patterns = ("*.jpg", "*.JPG", "*.jpeg", "*.png")

    for idx, class_name in enumerate(CLASS_NAMES):
        class_dir = data_dir / class_name
        if not class_dir.exists():
            continue

        class_images: List[Path] = []
        for pattern in patterns:
            class_images.extend(sorted(class_dir.glob(pattern)))

        image_paths.extend(str(path) for path in class_images)
        labels.extend([idx] * len(class_images))

    if not image_paths:
        raise ValueError(f"Veri seti bulunamadi: {data_dir}")

    return np.array(image_paths), np.array(labels, dtype=np.int32)


def split_test_set(
    image_paths: np.ndarray,
    labels: np.ndarray,
    train_split: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(image_paths))
    split_idx = int(len(image_paths) * train_split)
    test_indices = indices[split_idx:]
    return image_paths[test_indices], labels[test_indices]


def load_keras_model(model_path: Path) -> tf.keras.Model:
    kwargs = {
        "custom_objects": get_custom_objects(),
        "compile": False,
    }
    try:
        return tf.keras.models.load_model(model_path, safe_mode=False, **kwargs)
    except TypeError:
        return tf.keras.models.load_model(model_path, **kwargs)


def predict_keras(
    model: tf.keras.Model,
    image_paths: np.ndarray,
    image_size: int,
    batch_size: int,
) -> np.ndarray:
    predictions: List[int] = []

    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        batch_images = np.stack(
            [preprocess_image(Path(path), image_size) for path in batch_paths],
            axis=0,
        ).astype(np.float32)

        outputs = model.predict(batch_images, verbose=0)
        preds = np.argmax(outputs, axis=1)
        predictions.extend(preds.tolist())

    return np.asarray(predictions, dtype=np.int32)


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


def predict_tflite(
    interpreter,
    input_detail: dict,
    output_detail: dict,
    image_paths: np.ndarray,
    image_size: int,
    warmup_steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    latencies_ms: List[float] = []
    predictions: List[int] = []

    warmup_steps = min(warmup_steps, len(image_paths))
    for idx in range(warmup_steps):
        arr = preprocess_image(Path(image_paths[idx]), image_size)
        arr = np.expand_dims(arr, axis=0)
        arr = quantize_input(arr, input_detail)
        interpreter.set_tensor(input_detail["index"], arr)
        interpreter.invoke()

    for image_path in image_paths:
        arr = preprocess_image(Path(image_path), image_size)
        arr = np.expand_dims(arr, axis=0)
        arr = quantize_input(arr, input_detail)

        interpreter.set_tensor(input_detail["index"], arr)
        start = time.perf_counter()
        interpreter.invoke()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        latencies_ms.append(elapsed_ms)

        output = interpreter.get_tensor(output_detail["index"])[0]
        output = dequantize_output(output, output_detail)
        predictions.append(int(np.argmax(output)))

    return np.asarray(predictions, dtype=np.int32), np.asarray(latencies_ms, dtype=np.float32)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> List[float]:
    scores: List[float] = []
    for class_id in range(len(CLASS_NAMES)):
        mask = y_true == class_id
        if not np.any(mask):
            scores.append(float("nan"))
        else:
            scores.append(float(np.mean(y_pred[mask] == y_true[mask])))
    return scores


@dataclass
class ValidationResult:
    keras_accuracy: float
    tflite_accuracy: float
    agreement: float
    gap_abs: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    model_size_mb: float
    sample_count: int
    mismatch_count: int


def evaluate(
    keras_predictions: np.ndarray,
    tflite_predictions: np.ndarray,
    test_labels: np.ndarray,
    latencies_ms: np.ndarray,
    tflite_path: Path,
) -> ValidationResult:
    keras_acc = accuracy(test_labels, keras_predictions)
    tflite_acc = accuracy(test_labels, tflite_predictions)
    agreement = accuracy(keras_predictions, tflite_predictions)
    gap_abs = abs(keras_acc - tflite_acc)
    mismatch_count = int(np.sum(keras_predictions != tflite_predictions))

    return ValidationResult(
        keras_accuracy=keras_acc,
        tflite_accuracy=tflite_acc,
        agreement=agreement,
        gap_abs=gap_abs,
        avg_latency_ms=float(np.mean(latencies_ms)),
        p50_latency_ms=float(np.percentile(latencies_ms, 50)),
        p95_latency_ms=float(np.percentile(latencies_ms, 95)),
        model_size_mb=tflite_path.stat().st_size / (1024 * 1024),
        sample_count=len(test_labels),
        mismatch_count=mismatch_count,
    )


def write_summary_csv(
    report_path: Path,
    result: ValidationResult,
    keras_per_class: List[float],
    tflite_per_class: List[float],
) -> None:
    rows = [
        ("sample_count", result.sample_count),
        ("keras_accuracy", f"{result.keras_accuracy:.6f}"),
        ("tflite_accuracy", f"{result.tflite_accuracy:.6f}"),
        ("accuracy_gap_abs", f"{result.gap_abs:.6f}"),
        ("keras_tflite_agreement", f"{result.agreement:.6f}"),
        ("mismatch_count", result.mismatch_count),
        ("tflite_model_size_mb", f"{result.model_size_mb:.4f}"),
        ("tflite_latency_avg_ms", f"{result.avg_latency_ms:.3f}"),
        ("tflite_latency_p50_ms", f"{result.p50_latency_ms:.3f}"),
        ("tflite_latency_p95_ms", f"{result.p95_latency_ms:.3f}"),
    ]

    for idx, class_name in enumerate(CLASS_NAMES):
        rows.append((f"keras_class_acc::{class_name}", f"{keras_per_class[idx]:.6f}"))
        rows.append((f"tflite_class_acc::{class_name}", f"{tflite_per_class[idx]:.6f}"))

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerows(rows)


def write_mismatch_csv(
    mismatch_path: Path,
    image_paths: np.ndarray,
    labels: np.ndarray,
    keras_predictions: np.ndarray,
    tflite_predictions: np.ndarray,
) -> None:
    mismatch_path.parent.mkdir(parents=True, exist_ok=True)

    with mismatch_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label", "keras_pred", "tflite_pred"])

        for i in range(len(image_paths)):
            if keras_predictions[i] != tflite_predictions[i]:
                writer.writerow(
                    [
                        image_paths[i],
                        CLASS_NAMES[int(labels[i])],
                        CLASS_NAMES[int(keras_predictions[i])],
                        CLASS_NAMES[int(tflite_predictions[i])],
                    ]
                )


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    default_keras = project_root / "checkpoints_tomato_1_5x_baseline" / "best_model.keras"
    default_tflite = project_root / "artifacts" / "tflite" / "checkpoints_tomato_1_5x_baseline_best_model_fp16.tflite"
    default_data = get_default_data_dir(project_root)
    default_report = project_root / "artifacts" / "tflite" / "validation_summary.csv"
    default_mismatch = project_root / "artifacts" / "tflite" / "validation_mismatches.csv"

    parser = argparse.ArgumentParser(description="Keras ve TFLite model ciktilarini karsilastirir.")
    parser.add_argument("--keras-model", type=Path, default=default_keras)
    parser.add_argument("--tflite-model", type=Path, default=default_tflite)
    parser.add_argument("--data-dir", type=Path, default=default_data)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-test-samples", type=int, default=1000)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--report-csv", type=Path, default=default_report)
    parser.add_argument("--mismatch-csv", type=Path, default=default_mismatch)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    keras_path = args.keras_model.resolve()
    tflite_path = args.tflite_model.resolve()
    data_dir = args.data_dir.resolve()

    if not keras_path.exists():
        raise FileNotFoundError(f"Keras modeli bulunamadi: {keras_path}")
    if not tflite_path.exists():
        raise FileNotFoundError(f"TFLite modeli bulunamadi: {tflite_path}")

    print(f"Veri seti yukleniyor: {data_dir}")
    image_paths, labels = collect_dataset(data_dir)
    test_images, test_labels = split_test_set(image_paths, labels, args.train_split, args.seed)

    if args.max_test_samples > 0:
        test_images = test_images[: args.max_test_samples]
        test_labels = test_labels[: args.max_test_samples]

    print(f"Test ornek sayisi: {len(test_images)}")

    print(f"Keras model yukleniyor: {keras_path}")
    keras_model = load_keras_model(keras_path)

    print("Keras tahminleri aliniyor...")
    keras_preds = predict_keras(keras_model, test_images, args.image_size, args.batch_size)

    print(f"TFLite model yukleniyor: {tflite_path}")
    interpreter, input_detail, output_detail = build_interpreter(tflite_path, args.num_threads)

    print("TFLite tahminleri ve latency olcumu basladi...")
    tflite_preds, latencies_ms = predict_tflite(
        interpreter,
        input_detail,
        output_detail,
        test_images,
        args.image_size,
        args.warmup_steps,
    )

    result = evaluate(keras_preds, tflite_preds, test_labels, latencies_ms, tflite_path)
    keras_per_class = per_class_accuracy(test_labels, keras_preds)
    tflite_per_class = per_class_accuracy(test_labels, tflite_preds)

    report_csv = args.report_csv.resolve()
    mismatch_csv = args.mismatch_csv.resolve()
    write_summary_csv(report_csv, result, keras_per_class, tflite_per_class)
    write_mismatch_csv(mismatch_csv, test_images, test_labels, keras_preds, tflite_preds)

    print("\nKarsilastirma tamamlandi.")
    print(f"Keras accuracy:   {result.keras_accuracy:.4f}")
    print(f"TFLite accuracy:  {result.tflite_accuracy:.4f}")
    print(f"Mutlak fark:      {result.gap_abs:.4f}")
    print(f"Anlasma orani:    {result.agreement:.4f}")
    print(f"Model boyutu:     {result.model_size_mb:.2f} MB")
    print(f"Ortalama latency: {result.avg_latency_ms:.2f} ms")
    print(f"P95 latency:      {result.p95_latency_ms:.2f} ms")
    print(f"Mismatch sayisi:  {result.mismatch_count}")
    print(f"Rapor:            {report_csv}")
    print(f"Mismatch:         {mismatch_csv}")


if __name__ == "__main__":
    main()
