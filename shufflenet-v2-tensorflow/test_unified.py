from shufflenetv2_keras import ChannelShuffleLayer, ChannelSplitLayer
def channel_shuffle(x, groups):
    height, width, in_channels = x.shape[1:]
    channels_per_group = in_channels // groups
    x = tf.reshape(x, [-1, height, width, groups, channels_per_group])
    x = tf.transpose(x, [0, 1, 2, 4, 3])
    x = tf.reshape(x, [-1, height, width, in_channels])
    return x
"""
ShuffleNet V2 Unified Test Script
Tüm modeller için tutarlı metrikler üretir:
- Accuracy
- Precision (weighted & macro & per-class)
- Recall (weighted & macro & per-class)
- F1-Score (weighted & macro & per-class)
- Confusion Matrix
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
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    f1_score
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
    '1.0x_aug': {
        'model_path': './checkpoints_tomato_1_0x_aug/best_model.keras',
        'output_dir': './checkpoints_tomato_1_0x_aug',
        'model_scale': 1.0
    },
    '0.5x': {
        'model_path': './checkpoints_tomato_0_5x_baseline/best_model.keras',
        'output_dir': './checkpoints_tomato_0_5x_baseline',
        'model_scale': 0.5
    },
    '0.5x_transfer': {
        'model_path': './checkpoints_tomato_0_5x_transfer/best_model.keras',
        'output_dir': './checkpoints_tomato_0_5x_transfer',
        'model_scale': 0.5
    },
    '0.5x_aug': {
        'model_path': './checkpoints_tomato_0_5x_aug/best_model.keras',
        'output_dir': './checkpoints_tomato_0_5x_aug',
        'model_scale': 0.5
    },
    '1.0x': {
        'model_path': './checkpoints_tomato_baseline/best_model.keras',
        'output_dir': './checkpoints_tomato_baseline',
        'model_scale': 1.0
    },
    '1.0x_transfer': {
        'model_path': './checkpoints_tomato_1_0x_transfer/best_model.keras',
        'output_dir': './checkpoints_tomato_1_0x_transfer',
        'model_scale': 1.0
    },
    '1.0x_transfer_gpu': {
        'model_path': './checkpoints_tomato_1_0x_transfer_gpu/best_model.keras',
        'output_dir': './checkpoints_tomato_1_0x_transfer_gpu',
        'model_scale': 1.0
    },
    '1.5x': {
        'model_path': './checkpoints_tomato_1_5x_baseline/best_model.keras',
        'output_dir': './checkpoints_tomato_1_5x_baseline',
        'model_scale': 1.5
    },
    '1.5x_aug': {
        'model_path': './checkpoints_tomato_1_5x_aug/best_model.keras',
        'output_dir': './checkpoints_tomato_1_5x_aug',
        'model_scale': 1.5
    },
    '2.0x': {
        'model_path': './checkpoints_tomato_2_0x_baseline/best_model.keras',
        'output_dir': './checkpoints_tomato_2_0x_baseline',
        'model_scale': 2.0
    },
    '2.0x_aug': {
        'model_path': './checkpoints_tomato_2_0x_aug/best_model.keras',
        'output_dir': './checkpoints_tomato_2_0x_aug',
        'model_scale': 2.0
    }
    ,
    '1.5x_transfer_gpu_noaug': {
        'model_path': './checkpoints_tomato_1_5x_transfer_gpu_noaug/best_model_layer.keras',
        'output_dir': './checkpoints_tomato_1_5x_transfer_gpu_noaug',
        'model_scale': 1.5
    }
}

# Sınıf isimleri (tam)
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
    
    # Shuffle with fixed seed for reproducibility
    np.random.seed(42)
    indices = np.random.permutation(len(all_images))
    all_images = np.array(all_images)[indices]
    all_labels = np.array(all_labels)[indices]
    
    # Train/Test split
    split_idx = int(len(all_images) * TRAIN_SPLIT)
    test_images = all_images[split_idx:]
    test_labels = all_labels[split_idx:]
    
    print(f"\nToplam görüntü: {len(all_images)}")
    print(f"Test seti: {len(test_images)} görüntü")
    
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
        'ChannelSplit': ChannelSplit,
        'channel_shuffle': channel_shuffle
    }
    
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, safe_mode=False)
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


def calculate_all_metrics(test_labels, predictions):
    """Tüm metrikleri hesapla"""
    metrics = {}
    
    # ===== OVERALL METRICS =====
    metrics['accuracy'] = accuracy_score(test_labels, predictions)
    
    # Weighted metrics
    metrics['precision_weighted'] = precision_score(test_labels, predictions, average='weighted', zero_division=0)
    metrics['recall_weighted'] = recall_score(test_labels, predictions, average='weighted', zero_division=0)
    metrics['f1_weighted'] = f1_score(test_labels, predictions, average='weighted', zero_division=0)
    
    # Macro metrics
    metrics['precision_macro'] = precision_score(test_labels, predictions, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(test_labels, predictions, average='macro', zero_division=0)
    metrics['f1_macro'] = f1_score(test_labels, predictions, average='macro', zero_division=0)
    
    # ===== PER-CLASS METRICS =====
    precision_per, recall_per, f1_per, support_per = precision_recall_fscore_support(
        test_labels, predictions, average=None, zero_division=0
    )
    
    # Per-class accuracy
    cm = confusion_matrix(test_labels, predictions)
    accuracy_per = cm.diagonal() / cm.sum(axis=1)
    
    metrics['precision_per_class'] = precision_per
    metrics['recall_per_class'] = recall_per
    metrics['f1_per_class'] = f1_per
    metrics['accuracy_per_class'] = accuracy_per
    metrics['support_per_class'] = support_per
    
    # Confusion matrix
    metrics['confusion_matrix'] = cm
    
    return metrics


def print_metrics(metrics, model_name):
    """Metrikleri yazdır"""
    print("\n" + "="*80)
    print(f"TEST SONUÇLARI - ShuffleNet V2 {model_name}")
    print("="*80)
    
    print(f"\n{'='*40}")
    print("GENEL METRİKLER")
    print(f"{'='*40}")
    print(f"  Accuracy:           {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"\n  Weighted Metrics:")
    print(f"    Precision:        {metrics['precision_weighted']:.4f}")
    print(f"    Recall:           {metrics['recall_weighted']:.4f}")
    print(f"    F1-Score:         {metrics['f1_weighted']:.4f}")
    print(f"\n  Macro Metrics:")
    print(f"    Precision:        {metrics['precision_macro']:.4f}")
    print(f"    Recall:           {metrics['recall_macro']:.4f}")
    print(f"    F1-Score:         {metrics['f1_macro']:.4f}")
    
    print(f"\n{'='*40}")
    print("SINIF BAZINDA METRİKLER")
    print(f"{'='*40}")
    print(f"{'Sınıf':<18} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>8}")
    print("-"*80)
    
    for i, name in enumerate(SHORT_CLASS_NAMES):
        print(f"{name:<18} {metrics['accuracy_per_class'][i]:>10.4f} "
              f"{metrics['precision_per_class'][i]:>10.4f} "
              f"{metrics['recall_per_class'][i]:>10.4f} "
              f"{metrics['f1_per_class'][i]:>10.4f} "
              f"{int(metrics['support_per_class'][i]):>8}")
    
    print("-"*80)
    print(f"{'ORTALAMA':<18} {np.mean(metrics['accuracy_per_class']):>10.4f} "
          f"{metrics['precision_macro']:>10.4f} "
          f"{metrics['recall_macro']:>10.4f} "
          f"{metrics['f1_macro']:>10.4f} "
          f"{int(sum(metrics['support_per_class'])):>8}")


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
        f.write(f"Accuracy,{metrics['accuracy']:.6f}\n")
        f.write(f"Precision (Weighted),{metrics['precision_weighted']:.6f}\n")
        f.write(f"Recall (Weighted),{metrics['recall_weighted']:.6f}\n")
        f.write(f"F1-Score (Weighted),{metrics['f1_weighted']:.6f}\n")
        f.write(f"Precision (Macro),{metrics['precision_macro']:.6f}\n")
        f.write(f"Recall (Macro),{metrics['recall_macro']:.6f}\n")
        f.write(f"F1-Score (Macro),{metrics['f1_macro']:.6f}\n")
        f.write("\n")
        
        f.write("=== PER-CLASS METRICS ===\n")
        f.write("Class,Accuracy,Precision,Recall,F1-Score,Support\n")
        for i, name in enumerate(SHORT_CLASS_NAMES):
            f.write(f"{name},{metrics['accuracy_per_class'][i]:.6f},"
                   f"{metrics['precision_per_class'][i]:.6f},"
                   f"{metrics['recall_per_class'][i]:.6f},"
                   f"{metrics['f1_per_class'][i]:.6f},"
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
    
    plt.figure(figsize=(14, 12))
    
    # Normalize edilmiş confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Annotasyonlar: gerçek sayı + yüzde
    annot = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0:
                annot[i, j] = f'{cm[i, j]}\n({cm_normalized[i, j]*100:.1f}%)'
            else:
                annot[i, j] = '0'
    
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues',
                xticklabels=SHORT_CLASS_NAMES,
                yticklabels=SHORT_CLASS_NAMES,
                cbar_kws={'label': 'Count'})
    
    plt.title(f'Confusion Matrix - ShuffleNet V2 {model_name}\nAccuracy: {metrics["accuracy"]*100:.2f}%', fontsize=14)
    plt.ylabel('Gerçek Sınıf', fontsize=12)
    plt.xlabel('Tahmin Edilen Sınıf', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Confusion matrix kaydedildi: {save_path}")


def plot_per_class_metrics(metrics, model_name, output_dir):
    """Sınıf bazlı tüm metrikleri görselleştir"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    x = np.arange(len(SHORT_CLASS_NAMES))
    width = 0.6
    colors = ['steelblue', 'darkorange', 'forestgreen', 'crimson']
    
    metric_data = [
        ('Accuracy', metrics['accuracy_per_class'], metrics['accuracy'], axes[0, 0]),
        ('Precision', metrics['precision_per_class'], metrics['precision_weighted'], axes[0, 1]),
        ('Recall', metrics['recall_per_class'], metrics['recall_weighted'], axes[1, 0]),
        ('F1-Score', metrics['f1_per_class'], metrics['f1_weighted'], axes[1, 1])
    ]
    
    for idx, (name, values, weighted_avg, ax) in enumerate(metric_data):
        bars = ax.bar(x, values, width, color=colors[idx], alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Değerleri bar üzerine yaz
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8, rotation=90)
        
        ax.set_ylabel(name, fontsize=11)
        ax.set_title(f'{name} per Class - {model_name}', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(SHORT_CLASS_NAMES, rotation=45, ha='right', fontsize=9)
        ax.set_ylim([0.85, 1.02])
        ax.axhline(y=weighted_avg, color='red', linestyle='--', linewidth=2, 
                   label=f"Weighted Avg: {weighted_avg:.4f}")
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle(f'ShuffleNet V2 {model_name} - Per-Class Metrics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'per_class_metrics.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Per-class metrics kaydedildi: {save_path}")


def create_comparison_table(all_results):
    """Tüm modellerin karşılaştırma tablosu"""
    print("\n" + "="*100)
    print("MODEL KARŞILAŞTIRMA TABLOSU")
    print("="*100)
    
    print(f"\n{'Model':<8} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} "
          f"{'P-Macro':>10} {'R-Macro':>10} {'F1-Macro':>10} {'Params':>12}")
    print("-"*100)
    
    for model_name, results in all_results.items():
        m = results['metrics']
        p = results['params']
        print(f"{model_name:<8} {m['accuracy']:>10.4f} {m['precision_weighted']:>10.4f} "
              f"{m['recall_weighted']:>10.4f} {m['f1_weighted']:>10.4f} "
              f"{m['precision_macro']:>10.4f} {m['recall_macro']:>10.4f} "
              f"{m['f1_macro']:>10.4f} {p:>12,}")
    
    print("-"*100)
    
    # CSV'ye kaydet
    comparison_path = './model_comparison.csv'
    with open(comparison_path, 'w', encoding='utf-8') as f:
        f.write("ShuffleNet V2 Baseline Model Comparison\n")
        f.write(f"Date,{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("Transfer Learning,DISABLED\n")
        f.write("Data Augmentation,DISABLED\n\n")
        
        f.write("Model,Accuracy,Precision_W,Recall_W,F1_W,Precision_M,Recall_M,F1_M,Parameters\n")
        
        for model_name, results in all_results.items():
            m = results['metrics']
            p = results['params']
            f.write(f"{model_name},{m['accuracy']:.6f},{m['precision_weighted']:.6f},"
                   f"{m['recall_weighted']:.6f},{m['f1_weighted']:.6f},"
                   f"{m['precision_macro']:.6f},{m['recall_macro']:.6f},"
                   f"{m['f1_macro']:.6f},{p}\n")
    
    print(f"\n📁 Karşılaştırma tablosu kaydedildi: {comparison_path}")


def plot_comparison_chart(all_results):
    """Modellerin karşılaştırma grafiği"""
    models = list(all_results.keys())
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Tüm metrikler karşılaştırma
    metrics_to_plot = {
        'Accuracy': [all_results[m]['metrics']['accuracy'] for m in models],
        'Precision': [all_results[m]['metrics']['precision_weighted'] for m in models],
        'Recall': [all_results[m]['metrics']['recall_weighted'] for m in models],
        'F1-Score': [all_results[m]['metrics']['f1_weighted'] for m in models],
    }
    
    x = np.arange(len(models))
    width = 0.2
    multiplier = 0
    colors = ['steelblue', 'darkorange', 'forestgreen', 'crimson']
    
    for (metric_name, values), color in zip(metrics_to_plot.items(), colors):
        offset = width * multiplier
        bars = axes[0].bar(x + offset, values, width, label=metric_name, color=color)
        axes[0].bar_label(bars, fmt='%.4f', fontsize=8, rotation=90, padding=3)
        multiplier += 1
    
    axes[0].set_ylabel('Score')
    axes[0].set_title('Weighted Metrics Comparison', fontsize=12)
    axes[0].set_xticks(x + width * 1.5)
    axes[0].set_xticklabels([f'ShuffleNet V2 {m}' for m in models])
    axes[0].legend(loc='lower right')
    axes[0].set_ylim([0.95, 1.02])
    axes[0].grid(axis='y', alpha=0.3)
    
    # 2. Macro metrikler
    macro_metrics = {
        'Precision': [all_results[m]['metrics']['precision_macro'] for m in models],
        'Recall': [all_results[m]['metrics']['recall_macro'] for m in models],
        'F1-Score': [all_results[m]['metrics']['f1_macro'] for m in models],
    }
    
    multiplier = 0
    for (metric_name, values), color in zip(macro_metrics.items(), colors[1:]):
        offset = width * multiplier
        bars = axes[1].bar(x + offset, values, width, label=metric_name, color=color)
        axes[1].bar_label(bars, fmt='%.4f', fontsize=8, rotation=90, padding=3)
        multiplier += 1
    
    axes[1].set_ylabel('Score')
    axes[1].set_title('Macro Metrics Comparison', fontsize=12)
    axes[1].set_xticks(x + width)
    axes[1].set_xticklabels([f'ShuffleNet V2 {m}' for m in models])
    axes[1].legend(loc='lower right')
    axes[1].set_ylim([0.95, 1.02])
    axes[1].grid(axis='y', alpha=0.3)
    
    # 3. Model boyutu
    params = [all_results[m]['params'] / 1e6 for m in models]
    colors_size = ['lightblue', 'lightsalmon', 'lightgreen']
    bars = axes[2].bar(models, params, color=colors_size, edgecolor='black', linewidth=1)
    axes[2].bar_label(bars, fmt='%.2fM', fontsize=10)
    axes[2].set_ylabel('Parameters (Millions)')
    axes[2].set_title('Model Size Comparison', fontsize=12)
    axes[2].set_xticklabels([f'ShuffleNet V2 {m}' for m in models])
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.suptitle('ShuffleNet V2 Baseline Models - Comprehensive Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('./model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Karşılaştırma grafiği kaydedildi: ./model_comparison.png")


def test_single_model(model_name, config, test_images, test_labels):
    """Tek bir modeli test et"""
    print(f"\n{'='*80}")
    print(f"Testing ShuffleNet V2 {model_name}")
    print(f"{'='*80}")
    
    # Model yükle
    model = load_model(config['model_path'])
    if model is None:
        return None
    
    # Değerlendir
    predictions, probs = evaluate_model(model, test_images, test_labels)
    
    # Metrikleri hesapla
    metrics = calculate_all_metrics(test_labels, predictions)
    
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
    parser = argparse.ArgumentParser(description='Test ShuffleNet V2 models with unified metrics')
    parser.add_argument('--model', type=str, default='all',
                       choices=['all', '0.5x', '0.5x_aug', '0.5x_transfer', '1.0x', '1.0x_aug', '1.0x_transfer', '1.0x_transfer_gpu', '1.5x', '1.5x_aug', '2.0x', '2.0x_aug', '1.5x_transfer_gpu_noaug'],
                       help='Which model to test (default: all)')
    args = parser.parse_args()
    
    print("="*80)
    print("ShuffleNet V2 Unified Test Suite")
    print("Metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix")
    print("="*80)
    
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
    
    # Karşılaştırma tablosu ve grafik (birden fazla model test edilmişse)
    if len(all_results) > 1:
        create_comparison_table(all_results)
        plot_comparison_chart(all_results)
    
    print("\n" + "="*80)
    print("✅ Test tamamlandı!")
    print("="*80)
    print("\nOluşturulan dosyalar:")
    for model_name in all_results.keys():
        output_dir = MODEL_CONFIGS[model_name]['output_dir']
        print(f"  {model_name}:")
        print(f"    - {output_dir}/test_results.csv")
        print(f"    - {output_dir}/confusion_matrix.png")
        print(f"    - {output_dir}/per_class_metrics.png")
    
    if len(all_results) > 1:
        print(f"\n  Karşılaştırma:")
        print(f"    - ./model_comparison.csv")
        print(f"    - ./model_comparison.png")


if __name__ == '__main__':
    main()
