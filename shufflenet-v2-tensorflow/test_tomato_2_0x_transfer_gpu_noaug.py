from shufflenetv2_keras import ChannelShuffleLayer, ChannelSplitLayer
"""
ShuffleNet V2 2.0x Transfer Learning Test Script for Tomato Dataset
Eğitilmiş modeli test eder ve detaylı metrikler üretir.
"""

import tensorflow as tf
import numpy as np
import os
import glob
from PIL import Image
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import seaborn as sns

# GPU ayarları
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Hiperparametreler
BATCH_SIZE = 16  # 2.0x model için küçültüldü (GPU bellek sınırı)
IMAGE_SIZE = 224
NUM_CLASSES = 10
TRAIN_SPLIT = 0.8

# Veri seti yolu
DATA_DIR = '/mnt/021630F41630E9F5/PROJECTS/torch/tomato'

MODEL_PATH = './checkpoints_tomato_2_0x_transfer_gpu_noaug/best_model.keras'
CSV_PATH = './checkpoints_tomato_2_0x_transfer_gpu_noaug/test_results.csv'

# Sınıf isimleri
CLASS_NAMES = [
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

SHORT_CLASS_NAMES = [
    'Bacterial_spot',
    'Early_blight',
    'Late_blight',
    'Leaf_Mold',
    'Septoria_leaf_spot',
    'Spider_mites',
    'Target_Spot',
    'Yellow_Leaf_Curl',
    'Mosaic_virus',
    'Healthy'
]

# ...existing code for ChannelShuffle, ChannelSplit, load_dataset, preprocess_image, split_data, load_batch, evaluate_model, print_results, save_results_to_csv, plot_confusion_matrix, plot_per_class_accuracy ...

if __name__ == '__main__':
    from test_tomato import (
        ChannelShuffle, ChannelSplit, load_dataset, preprocess_image, split_data, load_batch,
        evaluate_model, print_results, save_results_to_csv, plot_confusion_matrix, plot_per_class_accuracy
    )
    import tensorflow as tf
    print("="*70)
    print("ShuffleNet V2 2.0x Transfer Learning - TEST")
    print("Transfer Learning: ENABLED")
    print("Data Augmentation: DISABLED")
    print("="*70)
    print(f"\nModel yükleniyor: {MODEL_PATH}")
    custom_objects = {
        'ChannelShuffle': ChannelShuffle,
        'ChannelSplit': ChannelSplit,
        'ChannelShuffleLayer': ChannelShuffleLayer,
        'ChannelSplitLayer': ChannelSplitLayer
    }
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
    print("✅ Model başarıyla yüklendi!")
    images, labels = load_dataset()
    print(f"\nToplam görüntü sayısı: {len(images)}")
    _, _, test_images, test_labels = split_data(images, labels, TRAIN_SPLIT, seed=42)
    print(f"Test seti: {len(test_images)} görüntü")
    unique, counts = np.unique(test_labels, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  {SHORT_CLASS_NAMES[u]}: {c}")
    predictions, probabilities = evaluate_model(model, test_images, test_labels)
    accuracy = print_results(test_labels, predictions, save_csv=True, csv_path=CSV_PATH)
    print("\nGörselleştirmeler oluşturuluyor...")
    plot_confusion_matrix(test_labels, predictions, './checkpoints_tomato_2_0x_transfer_gpu_noaug/confusion_matrix.png')
    plot_per_class_accuracy(test_labels, predictions, './checkpoints_tomato_2_0x_transfer_gpu_noaug/per_class_accuracy.png')
    print("\n" + "="*70)
    print("TEST TAMAMLANDI!")
    print("="*70)
    print(f"📊 Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"📁 Sonuçlar: ./checkpoints_tomato_2_0x_transfer_gpu_noaug/")
    print("="*70)
