# Android Kotlin Inference Reference

This folder contains a minimal Kotlin reference for on-device tomato disease inference.

## Included files

- TomatoClasses.kt: label list and inference result models
- ImagePreprocessor.kt: Bitmap -> normalized float tensor values
- TomatoClassifier.kt: TensorFlow Lite interpreter wrapper

## Required dependency

```kotlin
implementation("org.tensorflow:tensorflow-lite:2.15.0")
```

## Asset requirements

Place these files in your Android module assets directory:

- checkpoints_tomato_1_5x_baseline_best_model_int8.tflite or checkpoints_tomato_1_5x_baseline_best_model_fp16.tflite
- labels_tomato_10.txt

Recommended Gradle option for model assets:

```kotlin
android {
    aaptOptions {
        noCompress += "tflite"
    }
}
```

## Usage

```kotlin
val classifier = TomatoClassifier(
    context = context,
    modelAssetName = "checkpoints_tomato_1_5x_baseline_best_model_int8.tflite",
    numThreads = 4
)

val result = classifier.classify(bitmap)
val topLabel = result.top1.label
val confidence = result.top1.confidence
val latencyMs = result.latencyMs

classifier.close()
```

## Preprocess contract

The Kotlin preprocessor is aligned with Python scripts:

- Resize to 224x224 RGB
- Normalize from [0,255] to [-1,1] with x/127.5 - 1.0

## Notes

- If you switch to full-int8 input/output export, this classifier still supports int8/uint8 input tensors.
- For production apps, call classify on a background dispatcher (for example Dispatchers.Default).
