#!/usr/bin/env python3
"""Export ShuffleNetV2 Keras models to TensorFlow Lite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

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


def collect_image_paths(data_dir: Path) -> List[Path]:
    image_paths: List[Path] = []
    patterns = ("*.jpg", "*.JPG", "*.jpeg", "*.png")
    for class_name in CLASS_NAMES:
        class_dir = data_dir / class_name
        if not class_dir.exists():
            continue
        for pattern in patterns:
            image_paths.extend(sorted(class_dir.glob(pattern)))
    return image_paths


def representative_dataset(
    data_dir: Path,
    image_size: int,
    sample_count: int,
    seed: int,
) -> Iterable[list[np.ndarray]]:
    image_paths = collect_image_paths(data_dir)
    if not image_paths:
        raise ValueError(f"Representative dataset bulunamadi: {data_dir}")

    rng = np.random.default_rng(seed)
    sampled = rng.choice(image_paths, size=min(sample_count, len(image_paths)), replace=False)

    for image_path in sampled:
        arr = preprocess_image(Path(image_path), image_size)
        arr = np.expand_dims(arr, axis=0).astype(np.float32)
        yield [arr]


def load_model(model_path: Path) -> tf.keras.Model:
    kwargs = {
        "custom_objects": get_custom_objects(),
        "compile": False,
    }
    try:
        return tf.keras.models.load_model(model_path, safe_mode=False, **kwargs)
    except TypeError:
        return tf.keras.models.load_model(model_path, **kwargs)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    default_model = project_root / "checkpoints_tomato_1_5x_baseline" / "best_model.keras"
    default_output = project_root / "artifacts" / "tflite"
    default_data = get_default_data_dir(project_root)

    parser = argparse.ArgumentParser(description="Keras modelini TFLite formatina donusturur.")
    parser.add_argument("--keras-model", type=Path, default=default_model)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--model-id", type=str, default="")
    parser.add_argument("--quantization", choices=["fp32", "fp16", "int8"], default="fp16")
    parser.add_argument("--data-dir", type=Path, default=default_data)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-representative-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-select-tf-ops", action="store_true")
    parser.add_argument("--enforce-full-int8", action="store_true")
    return parser.parse_args()


def derive_model_id(model_path: Path) -> str:
    parent = model_path.parent.name
    stem = model_path.stem
    if parent:
        return f"{parent}_{stem}"
    return stem


def write_labels_file(output_dir: Path) -> Path:
    labels_path = output_dir / "labels_tomato_10.txt"
    labels_path.write_text("\n".join(CLASS_NAMES) + "\n", encoding="utf-8")
    return labels_path


def build_converter(
    model: tf.keras.Model,
    quantization: str,
    data_dir: Path,
    image_size: int,
    num_representative_samples: int,
    seed: int,
    allow_select_tf_ops: bool,
    enforce_full_int8: bool,
) -> tf.lite.TFLiteConverter:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if allow_select_tf_ops:
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]

    if quantization == "fp16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif quantization == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = lambda: representative_dataset(
            data_dir=data_dir,
            image_size=image_size,
            sample_count=num_representative_samples,
            seed=seed,
        )
        if enforce_full_int8:
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8

    return converter


def main() -> None:
    args = parse_args()

    keras_model_path = args.keras_model.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not keras_model_path.exists():
        raise FileNotFoundError(f"Model dosyasi bulunamadi: {keras_model_path}")

    model_id = args.model_id.strip() or derive_model_id(keras_model_path)
    tflite_name = f"{model_id}_{args.quantization}.tflite"
    tflite_path = output_dir / tflite_name

    print(f"Model yukleniyor: {keras_model_path}")
    model = load_model(keras_model_path)
    print(f"Parametre sayisi: {model.count_params():,}")

    converter = build_converter(
        model=model,
        quantization=args.quantization,
        data_dir=args.data_dir.resolve(),
        image_size=args.image_size,
        num_representative_samples=args.num_representative_samples,
        seed=args.seed,
        allow_select_tf_ops=args.allow_select_tf_ops,
        enforce_full_int8=args.enforce_full_int8,
    )

    print(f"Donusum basladi: quantization={args.quantization}")
    tflite_bytes = converter.convert()
    tflite_path.write_bytes(tflite_bytes)

    labels_path = write_labels_file(output_dir)

    metadata = {
        "keras_model": str(keras_model_path),
        "tflite_model": str(tflite_path),
        "labels": str(labels_path),
        "quantization": args.quantization,
        "image_size": args.image_size,
        "num_classes": len(CLASS_NAMES),
        "normalize": "(x/255 - 0.5) / 0.5",
    }
    metadata_path = output_dir / f"{model_id}_{args.quantization}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    size_mb = tflite_path.stat().st_size / (1024 * 1024)
    print("Donusum tamamlandi.")
    print(f"TFLite modeli: {tflite_path}")
    print(f"Model boyutu: {size_mb:.2f} MB")
    print(f"Etiket dosyasi: {labels_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
