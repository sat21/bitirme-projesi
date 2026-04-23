#!/usr/bin/env python3
"""Compare Keras and TFLite predictions on a single image."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse
from urllib.request import urlretrieve

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

INVALID_IMAGE_CONFIDENCE_THRESHOLD = 0.50
CONFIDENT_DIAGNOSIS_THRESHOLD = 0.75
MIN_MARGIN_THRESHOLD = 0.20


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


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    default_keras = project_root / "checkpoints_tomato_1_5x_baseline" / "best_model.keras"
    default_tflite = project_root / "artifacts" / "tflite" / "checkpoints_tomato_1_5x_baseline_best_model_int8.tflite"

    parser = argparse.ArgumentParser(description="Tek goruntude Keras ve TFLite sonucunu karsilastirir.")
    parser.add_argument("--keras-model", type=Path, default=default_keras)
    parser.add_argument("--tflite-model", type=Path, default=default_tflite)
    parser.add_argument("--image-path", type=Path)
    parser.add_argument("--image-url", type=str)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - np.max(x)
    exp_x = np.exp(x)
    s = np.sum(exp_x)
    if s == 0:
        return np.zeros_like(exp_x, dtype=np.float64)
    return exp_x / s


def ensure_probabilities(vector: np.ndarray) -> np.ndarray:
    vector = vector.astype(np.float64)
    if np.all(vector >= 0.0) and np.all(vector <= 1.0) and np.isclose(np.sum(vector), 1.0, atol=1e-3):
        return vector
    return softmax(vector)


def preprocess_image(image_path: Path, image_size: int) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    img = img.resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return np.expand_dims(arr, axis=0).astype(np.float32)


def load_keras_model(model_path: Path) -> tf.keras.Model:
    kwargs = {
        "custom_objects": get_custom_objects(),
        "compile": False,
    }
    try:
        return tf.keras.models.load_model(model_path, safe_mode=False, **kwargs)
    except TypeError:
        return tf.keras.models.load_model(model_path, **kwargs)


def keras_predict(model: tf.keras.Model, image_batch: np.ndarray) -> np.ndarray:
    output = model.predict(image_batch, verbose=0)[0]
    return ensure_probabilities(output)


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


def tflite_predict(
    interpreter,
    input_detail: dict,
    output_detail: dict,
    image_batch: np.ndarray,
) -> np.ndarray:
    input_batch = quantize_input(image_batch, input_detail)
    interpreter.set_tensor(input_detail["index"], input_batch)
    interpreter.invoke()
    raw_output = interpreter.get_tensor(output_detail["index"])[0]
    dequantized_output = dequantize_output(raw_output, output_detail)
    return ensure_probabilities(dequantized_output)


def format_label(raw: str) -> str:
    return raw.replace("Tomato___", "").replace("_", " ")


def topk(probabilities: np.ndarray, k: int) -> list[tuple[str, float]]:
    k = max(1, min(k, len(probabilities)))
    indices = np.argsort(probabilities)[::-1][:k]
    return [(CLASS_NAMES[idx], float(probabilities[idx])) for idx in indices]


def decision_from_probs(probabilities: np.ndarray) -> Tuple[str, float, float]:
    sorted_probs = np.sort(probabilities)[::-1]
    top1 = float(sorted_probs[0])
    top2 = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
    margin = top1 - top2

    if top1 < INVALID_IMAGE_CONFIDENCE_THRESHOLD:
        return "INVALID_IMAGE", top1, margin
    if top1 < CONFIDENT_DIAGNOSIS_THRESHOLD or margin < MIN_MARGIN_THRESHOLD:
        return "UNCERTAIN", top1, margin
    return "DIAGNOSIS", top1, margin


def maybe_download_image(image_url: Optional[str]) -> tuple[Optional[Path], Optional[tempfile.TemporaryDirectory]]:
    if not image_url:
        return None, None

    parsed = urlparse(image_url)
    suffix = Path(parsed.path).suffix or ".jpg"

    temp_dir = tempfile.TemporaryDirectory(prefix="tomatech_cmp_")
    out_path = Path(temp_dir.name) / f"downloaded{suffix}"
    urlretrieve(image_url, out_path)
    return out_path, temp_dir


def main() -> None:
    args = parse_args()

    if args.image_path is None and not args.image_url:
        raise ValueError("--image-path veya --image-url vermelisiniz.")

    keras_model_path = args.keras_model.resolve()
    tflite_model_path = args.tflite_model.resolve()

    if not keras_model_path.exists():
        raise FileNotFoundError(f"Keras model bulunamadi: {keras_model_path}")
    if not tflite_model_path.exists():
        raise FileNotFoundError(f"TFLite model bulunamadi: {tflite_model_path}")

    temp_dir_handle = None
    image_path: Optional[Path] = args.image_path.resolve() if args.image_path else None

    if image_path is None:
        image_path, temp_dir_handle = maybe_download_image(args.image_url)

    if image_path is None or not image_path.exists():
        raise FileNotFoundError("Gecerli bir goruntu dosyasi bulunamadi.")

    print(f"Goruntu: {image_path}")
    image_batch = preprocess_image(image_path, args.image_size)

    print(f"Keras model yukleniyor: {keras_model_path}")
    keras_model = load_keras_model(keras_model_path)
    keras_probs = keras_predict(keras_model, image_batch)

    print(f"TFLite model yukleniyor: {tflite_model_path}")
    interpreter, input_detail, output_detail = build_interpreter(tflite_model_path, args.num_threads)
    tflite_probs = tflite_predict(interpreter, input_detail, output_detail, image_batch)

    keras_topk = topk(keras_probs, args.top_k)
    tflite_topk = topk(tflite_probs, args.top_k)

    keras_decision, keras_top1_conf, keras_margin = decision_from_probs(keras_probs)
    tflite_decision, tflite_top1_conf, tflite_margin = decision_from_probs(tflite_probs)

    top1_same = keras_topk[0][0] == tflite_topk[0][0]
    prob_l1 = float(np.sum(np.abs(keras_probs - tflite_probs)))

    print("\n=== KERAS TOP-K ===")
    for idx, (label, prob) in enumerate(keras_topk, start=1):
        print(f"{idx}. {format_label(label):35s} {prob:.4f}")

    print("\n=== TFLITE TOP-K ===")
    for idx, (label, prob) in enumerate(tflite_topk, start=1):
        print(f"{idx}. {format_label(label):35s} {prob:.4f}")

    print("\n=== KARAR KATMANI ===")
    print(f"Keras  : {keras_decision} (top1={keras_top1_conf:.4f}, margin={keras_margin:.4f})")
    print(f"TFLite : {tflite_decision} (top1={tflite_top1_conf:.4f}, margin={tflite_margin:.4f})")

    print("\n=== KARSILASTIRMA ===")
    print(f"Top-1 ayni mi?      : {top1_same}")
    print(f"Prob L1 mesafesi    : {prob_l1:.6f}")
    print(f"Keras top1          : {format_label(keras_topk[0][0])}")
    print(f"TFLite top1         : {format_label(tflite_topk[0][0])}")

    output = {
        "image": str(image_path),
        "keras_topk": [{"label": label, "confidence": conf} for label, conf in keras_topk],
        "tflite_topk": [{"label": label, "confidence": conf} for label, conf in tflite_topk],
        "keras_decision": {
            "status": keras_decision,
            "top1_confidence": keras_top1_conf,
            "margin": keras_margin,
        },
        "tflite_decision": {
            "status": tflite_decision,
            "top1_confidence": tflite_top1_conf,
            "margin": tflite_margin,
        },
        "top1_same": top1_same,
        "probability_l1_distance": prob_l1,
    }

    if args.output_json:
        out_json = args.output_json.resolve()
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(f"\nJSON rapor: {out_json}")

    if temp_dir_handle is not None:
        temp_dir_handle.cleanup()


if __name__ == "__main__":
    main()
