"""
ShuffleNet V2 All Models Test Script
Tüm baseline modelleri (1.0x, 1.5x, 2.0x) test eder ve karşılaştırmalı metrikler üretir.

Metrics:
- Accuracy
- Precision (weighted & macro)
- Recall (weighted & macro)
- F1-Score (weighted & macro)
- Confusion Matrix
- Per-class metrics
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
import pandas as pd
from datetime import datetime
import argparse

# GPU ayarları
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Hiperparametreler
BATCH_SIZE = 32
IMAGE_SIZE = 224
NUM_CLASSES = 10
TRAIN_SPLIT = 0.8

# Veri seti yolu
DATA_DIR = '/mnt/021630F41630E9F5/PROJECTS/torch/tomato'

# Model yapılandırmaları
MODEL_CONFIGS = {
    '1.0x': {
        'model_path': './checkpoints_tomato_baseline/best_model.keras',
        'output_dir': './checkpoints_tomato_baseline',
        'model_scale': 1.0
    },
    '1.5x': {
        'model_path': './checkpoints_tomato_1_5x_baseline/best_model.keras',
        'output_dir': './checkpoints_tomato_1_5x_baseline',
        'model_scale': 1.5
    },
    '2.0x': {
        'model_path': './checkpoints_tomato_2_0x_baseline/best_model.keras',
        'output_dir': './checkpoints_tomato_2_0x_baseline',
        'model_scale': 2.0
    }
}

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

# Kısa sınıf isimleri (görselleştirme için)
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


# ==================== Custom Layers ====================

class ChannelShuffle(tf.keras.layers.Layer):
    """Channel shuffle layer for ShuffleNet"""
    def __init__(self, groups=2, **kwargs):
        super(ChannelShuffle, self).__init__(**kwargs)
        self.groups = groups
    
    def call(self, x):
        shape = tf.shape(x)
        batch_size = shape[0]
        height = shape[1]
        width = shape[2]
        channels = x.shape[-1]
        channels_per_group = channels // self.groups
        
        x = tf.reshape(x, [batch_size, height, width, self.groups, channels_per_group])
        x = tf.transpose(x, [0, 1, 2, 4, 3])
        x = tf.reshape(x, [batch_size, height, width, channels])
        
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({'groups': self.groups})
        return config


class ChannelSplit(tf.keras.layers.Layer):
    """Channel split layer"""
    def __init__(self, split_idx, **kwargs):
        super(ChannelSplit, self).__init__(**kwargs)
        self.split_idx = split_idx
    
    def call(self, x):
        if self.split_idx == 0:
            return x[:, :, :, :x.shape[-1]//2]
        else:
            return x[:, :, :, x.shape[-1]//2:]
    
    def get_config(self):
        config = super().get_config()
        config.update({'split_idx': self.split_idx})
        return config


# ==================== Data Loading ====================

def load_dataset():
    """Veri setini yükle ve train/test olarak ayır"""
    print("\nVeri seti yükleniyor...")
    
    all_images = []
    all_labels = []
    
    for idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(DATA_DIR, class_name)
        if not os.path.exists(class_dir):
            print(f"  Uyarı: {class_dir} bulunamadı!")
            continue
            
        images = glob.glob(os.path.join(class_dir, '*.jpg')) + \
                 glob.glob(os.path.join(class_dir, '*.JPG')) + \
                 glob.glob(os.path.join(class_dir, '*.jpeg')) + \
                 glob.glob(os.path.join(class_dir, '*.png'))
        
        all_images.extend(images)
        all_labels.extend([idx] * len(images))
        print(f"  {SHORT_CLASS_NAMES[idx]}: {len(images)} görüntü")
    
    # Shuffle
    np.random.seed(42)
    indices = np.random.permutation(len(all_images))
    all_images = np.array(all_images)[indices]
    all_labels = np.array(all_labels)[indices]
    
    # Train/Test split
    split_idx = int(len(all_images) * TRAIN_SPLIT)
    test_images = all_images[split_idx:]
    test_labels = all_labels[split_idx:]
    
    print(f"\nTest seti: {len(test_images)} görüntü")
    
    return test_images, test_labels


def preprocess_image(img_path):
    """Görüntüyü ön işle"""
    img = Image.open(img_path).convert('RGB')
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    img = np.array(img, dtype=np.float32) / 255.0
    img = (img - 0.5) / 0.5  # Normalize [-1, 1]
    return img


def load_batch(image_paths):
    """Bir batch görüntü yükle"""
    images = []
    for path in image_paths:
        img = preprocess_image(path)
        images.append(img)
    return np.array(images)


# ==================== Model Loading ====================

def load_model(model_path):
    """Modeli yükle"""
    print(f"\nModel yükleniyor: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"  ❌ Model dosyası bulunamadı: {model_path}")
        return None
    
    custom_objects = {
        'ChannelShuffle': ChannelShuffle,
        'ChannelSplit': ChannelSplit
    }
    
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    print(f"  ✅ Model yüklendi: {model.count_params():,} parametre")
    
    return model


# ==================== Evaluation ====================

def evaluate_model(model, test_images, test_labels):
    """Model değerlendirmesi yap"""
    print("\nModel değerlendiriliyor...")
    
    all_predictions = []
    all_probs = []
    
    num_batches = (len(test_images) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in range(num_batches):
        start_idx = i * BATCH_SIZE
        end_idx = min((i + 1) * BATCH_SIZE, len(test_images))
        batch_paths = test_images[start_idx:end_idx]
        
        batch_images = load_batch(batch_paths)
        logits = model.predict(batch_images, verbose=0)
        probs = tf.nn.softmax(logits).numpy()
        preds = np.argmax(logits, axis=1)
        
        all_predictions.extend(preds)
        all_probs.extend(probs)
        
        print(f"\r  İlerleme: {end_idx}/{len(test_images)}", end="")
    
    print()
    
    return np.array(all_predictions), np.array(all_probs)


def calculate_metrics(test_labels, predictions):
    """Tüm metrikleri hesapla"""
    metrics = {}
    
    # Accuracy
    metrics['accuracy'] = accuracy_score(test_labels, predictions)
    
    # Weighted metrics
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        test_labels, predictions, average='weighted'
    )
    metrics['precision_weighted'] = precision_w
    metrics['recall_weighted'] = recall_w
    metrics['f1_weighted'] = f1_w
    
    # Macro metrics
    precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(
        test_labels, predictions, average='macro'
    )
    metrics['precision_macro'] = precision_m
    metrics['recall_macro'] = recall_m
    metrics['f1_macro'] = f1_m
    
    # Per-class metrics
    precision_per, recall_per, f1_per, support_per = precision_recall_fscore_support(
        test_labels, predictions, average=None
    )
    metrics['precision_per_class'] = precision_per
    metrics['recall_per_class'] = recall_per
    metrics['f1_per_class'] = f1_per
    metrics['support_per_class'] = support_per
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(test_labels, predictions)
    
    return metrics


def print_metrics(metrics, model_name):
    """Metrikleri yazdır"""
    print("\n" + "="*70)
    print(f"TEST SONUÇLARI - {model_name}")
    print("="*70)
    
    print(f"\n📊 Genel Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    
    print(f"\n📈 Weighted Metrics:")
    print(f"   Precision: {metrics['precision_weighted']:.4f}")
    print(f"   Recall:    {metrics['recall_weighted']:.4f}")
    print(f"   F1-Score:  {metrics['f1_weighted']:.4f}")
    
    print(f"\n📈 Macro Metrics:")
    print(f"   Precision: {metrics['precision_macro']:.4f}")
    print(f"   Recall:    {metrics['recall_macro']:.4f}")
    print(f"   F1-Score:  {metrics['f1_macro']:.4f}")
    
    print("\n" + "-"*70)
    print("SINIF BAZINDA METRIKLER")
    print("-"*70)
    print(f"{'Sınıf':<20} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("-"*70)
    
    for i, name in enumerate(SHORT_CLASS_NAMES):
        print(f"{name:<20} {metrics['precision_per_class'][i]:>10.4f} "
              f"{metrics['recall_per_class'][i]:>10.4f} "
              f"{metrics['f1_per_class'][i]:>10.4f} "
              f"{int(metrics['support_per_class'][i]):>10}")


def save_results_to_csv(metrics, model_name, output_dir):
    """Test sonuçlarını CSV dosyasına kaydet"""
    csv_path = os.path.join(output_dir, 'test_results.csv')
    
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write(f"ShuffleNet V2 {model_name} Baseline Test Results\n")
        f.write(f"Date,{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Transfer Learning,DISABLED\n")
        f.write(f"Data Augmentation,DISABLED\n")
        f.write(f"Total Test Samples,{int(sum(metrics['support_per_class']))}\n")
        f.write("\n")
        
        f.write("=== OVERALL METRICS ===\n")
        f.write("Metric,Value\n")
        f.write(f"Accuracy,{metrics['accuracy']:.4f}\n")
        f.write(f"Weighted Precision,{metrics['precision_weighted']:.4f}\n")
        f.write(f"Weighted Recall,{metrics['recall_weighted']:.4f}\n")
        f.write(f"Weighted F1-Score,{metrics['f1_weighted']:.4f}\n")
        f.write(f"Macro Precision,{metrics['precision_macro']:.4f}\n")
        f.write(f"Macro Recall,{metrics['recall_macro']:.4f}\n")
        f.write(f"Macro F1-Score,{metrics['f1_macro']:.4f}\n")
        f.write("\n")
        
        f.write("=== PER-CLASS METRICS ===\n")
        f.write("Class,Precision,Recall,F1-Score,Support\n")
        for i, name in enumerate(SHORT_CLASS_NAMES):
            f.write(f"{name},{metrics['precision_per_class'][i]:.4f},"
                   f"{metrics['recall_per_class'][i]:.4f},"
                   f"{metrics['f1_per_class'][i]:.4f},"
                   f"{int(metrics['support_per_class'][i])}\n")
        f.write("\n")
        
        f.write("=== CONFUSION MATRIX ===\n")
        cm = metrics['confusion_matrix']
        f.write("," + ",".join(SHORT_CLASS_NAMES) + "\n")
        for i, name in enumerate(SHORT_CLASS_NAMES):
            f.write(name + "," + ",".join(map(str, cm[i])) + "\n")
    
    print(f"\n📁 Sonuçlar kaydedildi: {csv_path}")


def plot_confusion_matrix(metrics, model_name, output_dir):
    """Confusion matrix görselleştir"""
    cm = metrics['confusion_matrix']
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=SHORT_CLASS_NAMES,
                yticklabels=SHORT_CLASS_NAMES)
    plt.title(f'Confusion Matrix - ShuffleNet V2 {model_name}')
    plt.ylabel('Gerçek Sınıf')
    plt.xlabel('Tahmin Edilen Sınıf')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"📊 Confusion matrix kaydedildi: {save_path}")


def plot_per_class_metrics(metrics, model_name, output_dir):
    """Sınıf bazlı metrikleri görselleştir"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    x = np.arange(len(SHORT_CLASS_NAMES))
    width = 0.6
    
    # Precision
    axes[0].bar(x, metrics['precision_per_class'], width, color='steelblue')
    axes[0].set_ylabel('Precision')
    axes[0].set_title(f'Precision per Class - {model_name}')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(SHORT_CLASS_NAMES, rotation=45, ha='right')
    axes[0].set_ylim([0.9, 1.01])
    axes[0].axhline(y=metrics['precision_weighted'], color='r', linestyle='--', label=f"Weighted: {metrics['precision_weighted']:.4f}")
    axes[0].legend()
    
    # Recall
    axes[1].bar(x, metrics['recall_per_class'], width, color='darkorange')
    axes[1].set_ylabel('Recall')
    axes[1].set_title(f'Recall per Class - {model_name}')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(SHORT_CLASS_NAMES, rotation=45, ha='right')
    axes[1].set_ylim([0.9, 1.01])
    axes[1].axhline(y=metrics['recall_weighted'], color='r', linestyle='--', label=f"Weighted: {metrics['recall_weighted']:.4f}")
    axes[1].legend()
    
    # F1-Score
    axes[2].bar(x, metrics['f1_per_class'], width, color='forestgreen')
    axes[2].set_ylabel('F1-Score')
    axes[2].set_title(f'F1-Score per Class - {model_name}')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(SHORT_CLASS_NAMES, rotation=45, ha='right')
    axes[2].set_ylim([0.9, 1.01])
    axes[2].axhline(y=metrics['f1_weighted'], color='r', linestyle='--', label=f"Weighted: {metrics['f1_weighted']:.4f}")
    axes[2].legend()
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'per_class_metrics.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"📊 Per-class metrics kaydedildi: {save_path}")


def create_comparison_table(all_results):
    """Tüm modellerin karşılaştırma tablosu"""
    print("\n" + "="*90)
    print("MODEL KARŞILAŞTIRMA TABLOSU")
    print("="*90)
    
    print(f"\n{'Model':<10} {'Accuracy':>10} {'Precision':>12} {'Recall':>10} {'F1-Score':>10} {'Params':>15}")
    print("-"*90)
    
    for model_name, results in all_results.items():
        metrics = results['metrics']
        params = results['params']
        print(f"{model_name:<10} {metrics['accuracy']:>10.4f} "
              f"{metrics['precision_weighted']:>12.4f} "
              f"{metrics['recall_weighted']:>10.4f} "
              f"{metrics['f1_weighted']:>10.4f} "
              f"{params:>15,}")
    
    print("-"*90)
    
    # CSV'ye kaydet
    comparison_path = './model_comparison.csv'
    with open(comparison_path, 'w', encoding='utf-8') as f:
        f.write("Model Comparison - ShuffleNet V2 Baseline Models\n")
        f.write(f"Date,{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Model,Accuracy,Precision_W,Recall_W,F1_W,Precision_M,Recall_M,F1_M,Parameters\n")
        
        for model_name, results in all_results.items():
            m = results['metrics']
            p = results['params']
            f.write(f"{model_name},{m['accuracy']:.4f},{m['precision_weighted']:.4f},"
                   f"{m['recall_weighted']:.4f},{m['f1_weighted']:.4f},"
                   f"{m['precision_macro']:.4f},{m['recall_macro']:.4f},"
                   f"{m['f1_macro']:.4f},{p}\n")
    
    print(f"\n📁 Karşılaştırma tablosu kaydedildi: {comparison_path}")


def plot_comparison_chart(all_results):
    """Modellerin karşılaştırma grafiği"""
    models = list(all_results.keys())
    
    metrics_to_plot = {
        'Accuracy': [all_results[m]['metrics']['accuracy'] for m in models],
        'Precision': [all_results[m]['metrics']['precision_weighted'] for m in models],
        'Recall': [all_results[m]['metrics']['recall_weighted'] for m in models],
        'F1-Score': [all_results[m]['metrics']['f1_weighted'] for m in models],
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart
    x = np.arange(len(models))
    width = 0.2
    multiplier = 0
    
    for metric_name, values in metrics_to_plot.items():
        offset = width * multiplier
        bars = axes[0].bar(x + offset, values, width, label=metric_name)
        axes[0].bar_label(bars, fmt='%.4f', fontsize=8)
        multiplier += 1
    
    axes[0].set_ylabel('Score')
    axes[0].set_title('Model Comparison - Metrics')
    axes[0].set_xticks(x + width * 1.5)
    axes[0].set_xticklabels(models)
    axes[0].legend(loc='lower right')
    axes[0].set_ylim([0.95, 1.01])
    
    # Parameter count
    params = [all_results[m]['params'] / 1e6 for m in models]  # Millions
    bars = axes[1].bar(models, params, color=['steelblue', 'darkorange', 'forestgreen'])
    axes[1].bar_label(bars, fmt='%.2fM', fontsize=10)
    axes[1].set_ylabel('Parameters (Millions)')
    axes[1].set_title('Model Size Comparison')
    
    plt.tight_layout()
    plt.savefig('./model_comparison.png', dpi=150)
    plt.close()
    print(f"📊 Karşılaştırma grafiği kaydedildi: ./model_comparison.png")


def test_single_model(model_name, config, test_images, test_labels):
    """Tek bir modeli test et"""
    print(f"\n{'='*70}")
    print(f"Testing ShuffleNet V2 {model_name}")
    print(f"{'='*70}")
    
    # Model yükle
    model = load_model(config['model_path'])
    if model is None:
        return None
    
    # Değerlendir
    predictions, probs = evaluate_model(model, test_images, test_labels)
    
    # Metrikleri hesapla
    metrics = calculate_metrics(test_labels, predictions)
    
    # Sonuçları yazdır
    print_metrics(metrics, model_name)
    
    # Kaydet
    os.makedirs(config['output_dir'], exist_ok=True)
    save_results_to_csv(metrics, model_name, config['output_dir'])
    plot_confusion_matrix(metrics, model_name, config['output_dir'])
    plot_per_class_metrics(metrics, model_name, config['output_dir'])
    
    return {
        'metrics': metrics,
        'params': model.count_params()
    }


def main():
    parser = argparse.ArgumentParser(description='Test ShuffleNet V2 models')
    parser.add_argument('--model', type=str, default='all',
                       choices=['all', '1.0x', '1.5x', '2.0x'],
                       help='Which model to test (default: all)')
    args = parser.parse_args()
    
    print("="*70)
    print("ShuffleNet V2 Model Test Suite")
    print("="*70)
    
    # Veri setini yükle
    test_images, test_labels = load_dataset()
    
    # Test edilecek modeller
    if args.model == 'all':
        models_to_test = MODEL_CONFIGS
    else:
        models_to_test = {args.model: MODEL_CONFIGS[args.model]}
    
    # Tüm sonuçları topla
    all_results = {}
    
    for model_name, config in models_to_test.items():
        result = test_single_model(model_name, config, test_images, test_labels)
        if result is not None:
            all_results[model_name] = result
    
    # Karşılaştırma tablosu ve grafik
    if len(all_results) > 1:
        create_comparison_table(all_results)
        plot_comparison_chart(all_results)
    
    print("\n" + "="*70)
    print("✅ Test tamamlandı!")
    print("="*70)


if __name__ == '__main__':
    main()
