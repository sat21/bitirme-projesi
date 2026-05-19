import tensorflow as tf
import os
from train_tomato_tf2 import ChannelShuffle, ChannelSplit

def convert_to_tflite():
    keras_model_path = "../checkpoints_tomato_baseline/best_model.keras"
    tflite_model_dir = "../checkpoints_tomato_baseline"
    tflite_model_path = os.path.join(tflite_model_dir, "best_model.tflite")
    
    # Just in case we run directly from root
    if not os.path.exists(keras_model_path):
        keras_model_path = "checkpoints_tomato_baseline/best_model.keras"
        tflite_model_path = "checkpoints_tomato_baseline/best_model.tflite"

    print(f"Keras modeli yükleniyor: {keras_model_path}...")
    model = tf.keras.models.load_model(keras_model_path, custom_objects={
        'ChannelShuffle': ChannelShuffle,
        'ChannelSplit': ChannelSplit
    })

    print("TFLite formatına dönüştürülüyor (FP32 - Doğruluk kaybı yok)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Herhangi bir kuantizasyon (kalibrasyon) yapmıyoruz, Orijinal FP32 (float32) hassasiyetinde kalacak.
    tflite_model = converter.convert()

    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)

    print(f"Dönüştürme başarıyla tamamlandı!")
    print(f"TFLite modeli şuraya kaydedildi: {tflite_model_path}")
    print(f"Model boyutu: {os.path.getsize(tflite_model_path) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    convert_to_tflite()
