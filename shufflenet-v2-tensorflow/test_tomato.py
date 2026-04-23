"""
ShuffleNet V2 1.0 Baseline Test Script for Tomato Dataset
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
BATCH_SIZE = 32
IMAGE_SIZE = 224
NUM_CLASSES = 10
TRAIN_SPLIT = 0.8

# Veri seti yolu
DATA_DIR = '/mnt/021630F41630E9F5/PROJECTS/torch/tomato'

# Model yolu
MODEL_PATH = './checkpoints_tomato_baseline/best_model.keras'

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
    """Veri setini yükle"""
    images = []
    labels = []
    
    print("Veri seti yükleniyor...")
    for idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(DATA_DIR, class_name)
        if not os.path.exists(class_dir):
            print(f"Uyarı: {class_dir} bulunamadı!")
            continue
            
        image_files = glob.glob(os.path.join(class_dir, '*.jpg')) + \
                      glob.glob(os.path.join(class_dir, '*.JPG')) + \
                      glob.glob(os.path.join(class_dir, '*.jpeg')) + \
                      glob.glob(os.path.join(class_dir, '*.png'))
        
        print(f"  {class_name}: {len(image_files)} görüntü")
        
        for img_path in image_files:
            images.append(img_path)
            labels.append(idx)
    
    return np.array(images), np.array(labels)


def preprocess_image(image_path):
    """Görüntüyü ön işle"""
    if isinstance(image_path, bytes):
        image_path = image_path.decode('utf-8')
    img = Image.open(image_path).convert('RGB')
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    img = np.array(img, dtype=np.float32)
    img = img / 255.0
    img = (img - 0.5) / 0.5
    return img


def split_data(images, labels, train_ratio=0.8, seed=42):
    """Eğitim ve test setlerine ayır (aynı seed ile tutarlılık sağla)"""
    np.random.seed(seed)
    indices = np.random.permutation(len(images))
    images = images[indices]
    labels = labels[indices]
    
    n_samples = len(images)
    n_train = int(n_samples * train_ratio)
    
    train_images = images[:n_train]
    train_labels = labels[:n_train]
    test_images = images[n_train:]
    test_labels = labels[n_train:]
    
    return train_images, train_labels, test_images, test_labels


def load_batch(image_paths):
    """Batch yükle"""
    batch_images = []
    for img_path in image_paths:
        img = preprocess_image(img_path)
        batch_images.append(img)
    return np.array(batch_images)


# ==================== Evaluation ====================

def evaluate_model(model, test_images, test_labels):
    """Model değerlendirmesi yap"""
    print("\nModel değerlendiriliyor...")
    
    all_predictions = []
    all_probs = []
    
    # Batch batch tahmin yap
    num_batches = (len(test_images) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in range(num_batches):
        start_idx = i * BATCH_SIZE
        end_idx = min((i + 1) * BATCH_SIZE, len(test_images))
        batch_paths = test_images[start_idx:end_idx]
        
        # Görüntüleri yükle
        batch_images = load_batch(batch_paths)
        
        # Tahmin
        logits = model.predict(batch_images, verbose=0)
        probs = tf.nn.softmax(logits).numpy()
        preds = np.argmax(logits, axis=1)
        
        all_predictions.extend(preds)
        all_probs.extend(probs)
        
        # İlerleme
        print(f"\r  İlerleme: {end_idx}/{len(test_images)}", end="")
    
    print()
    
    return np.array(all_predictions), np.array(all_probs)


def print_results(test_labels, predictions, save_csv=True, csv_path='checkpoints_tomato_baseline/test_results.csv'):
    """Sonuçları yazdır ve CSV'ye kaydet"""
    print("\n" + "="*70)
    print("TEST SONUÇLARI")
    print("="*70)
    
    # Genel accuracy
    accuracy = accuracy_score(test_labels, predictions)
    print(f"\n📊 Genel Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Precision, Recall, F1
    precision, recall, f1, support = precision_recall_fscore_support(
        test_labels, predictions, average='weighted'
    )
    print(f"\n📈 Weighted Metrics:")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    
    # Macro metrics
    precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(
        test_labels, predictions, average='macro'
    )
    print(f"\n📈 Macro Metrics:")
    print(f"   Precision: {precision_m:.4f}")
    print(f"   Recall:    {recall_m:.4f}")
    print(f"   F1-Score:  {f1_m:.4f}")
    
    # Her sınıf için detaylı rapor
    print("\n" + "-"*70)
    print("SINIF BAZINDA DETAYLI RAPOR")
    print("-"*70)
    report = classification_report(
        test_labels, predictions, 
        target_names=SHORT_CLASS_NAMES,
        digits=4
    )
    print(report)
    
    # CSV'ye kaydet
    if save_csv:
        save_results_to_csv(test_labels, predictions, accuracy, csv_path)
    
    return accuracy


def save_results_to_csv(test_labels, predictions, accuracy, csv_path):
    """Test sonuçlarını CSV dosyasına kaydet"""
    import csv
    from datetime import datetime
    
    # Sınıf bazlı metrikler
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        test_labels, predictions, average=None
    )
    
    # Weighted ve Macro metrikler
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        test_labels, predictions, average='weighted'
    )
    precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(
        test_labels, predictions, average='macro'
    )
    
    # Confusion matrix
    cm = confusion_matrix(test_labels, predictions)
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header bilgileri
        writer.writerow(['ShuffleNet V2 1.0 Baseline Test Results'])
        writer.writerow(['Date', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
        writer.writerow(['Transfer Learning', 'DISABLED'])
        writer.writerow(['Data Augmentation', 'DISABLED'])
        writer.writerow(['Total Test Samples', len(test_labels)])
        writer.writerow([])
        
        # Genel metrikler
        writer.writerow(['=== OVERALL METRICS ==='])
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Accuracy', f'{accuracy:.4f}'])
        writer.writerow(['Weighted Precision', f'{precision_w:.4f}'])
        writer.writerow(['Weighted Recall', f'{recall_w:.4f}'])
        writer.writerow(['Weighted F1-Score', f'{f1_w:.4f}'])
        writer.writerow(['Macro Precision', f'{precision_m:.4f}'])
        writer.writerow(['Macro Recall', f'{recall_m:.4f}'])
        writer.writerow(['Macro F1-Score', f'{f1_m:.4f}'])
        writer.writerow([])
        
        # Sınıf bazlı metrikler
        writer.writerow(['=== PER-CLASS METRICS ==='])
        writer.writerow(['Class', 'Precision', 'Recall', 'F1-Score', 'Support'])
        for i, class_name in enumerate(SHORT_CLASS_NAMES):
            writer.writerow([
                class_name,
                f'{precision_per_class[i]:.4f}',
                f'{recall_per_class[i]:.4f}',
                f'{f1_per_class[i]:.4f}',
                int(support_per_class[i])
            ])
        writer.writerow([])
        
        # Confusion Matrix
        writer.writerow(['=== CONFUSION MATRIX ==='])
        writer.writerow([''] + SHORT_CLASS_NAMES)
        for i, class_name in enumerate(SHORT_CLASS_NAMES):
            writer.writerow([class_name] + [str(x) for x in cm[i]])
    
    print(f"✅ Test sonuçları CSV'ye kaydedildi: {csv_path}")


def plot_confusion_matrix(test_labels, predictions, save_path='confusion_matrix.png'):
    """Confusion matrix görselleştir"""
    cm = confusion_matrix(test_labels, predictions)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=SHORT_CLASS_NAMES,
        yticklabels=SHORT_CLASS_NAMES
    )
    plt.title('Confusion Matrix - ShuffleNet V2 Baseline\n(No Transfer Learning, No Data Augmentation)', fontsize=14)
    plt.xlabel('Tahmin Edilen Sınıf', fontsize=12)
    plt.ylabel('Gerçek Sınıf', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n✅ Confusion matrix kaydedildi: {save_path}")


def plot_per_class_accuracy(test_labels, predictions, save_path='per_class_accuracy.png'):
    """Her sınıf için accuracy bar chart"""
    cm = confusion_matrix(test_labels, predictions)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(NUM_CLASSES), per_class_acc, color='steelblue', edgecolor='black')
    
    # Her bar üzerine değer yaz
    for bar, acc in zip(bars, per_class_acc):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.xticks(range(NUM_CLASSES), SHORT_CLASS_NAMES, rotation=45, ha='right')
    plt.xlabel('Sınıf', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Sınıf Bazında Accuracy - ShuffleNet V2 Baseline', fontsize=14)
    plt.ylim(0, 1.1)
    plt.axhline(y=np.mean(per_class_acc), color='red', linestyle='--', label=f'Ortalama: {np.mean(per_class_acc):.2f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ Per-class accuracy grafiği kaydedildi: {save_path}")


# ==================== Main ====================

def main():
    print("="*70)
    print("ShuffleNet V2 1.0 Baseline - TEST")
    print("Transfer Learning: DISABLED")
    print("Data Augmentation: DISABLED")
    print("="*70)
    
    # Model yükle
    print(f"\nModel yükleniyor: {MODEL_PATH}")
    
    custom_objects = {
        'ChannelShuffle': ChannelShuffle,
        'ChannelSplit': ChannelSplit
    }
    
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
    print("✅ Model başarıyla yüklendi!")
    
    # Veri setini yükle
    images, labels = load_dataset()
    print(f"\nToplam görüntü sayısı: {len(images)}")
    
    # Aynı split'i uygula (seed ile tutarlılık)
    _, _, test_images, test_labels = split_data(images, labels, TRAIN_SPLIT, seed=42)
    print(f"Test seti: {len(test_images)} görüntü")
    
    # Sınıf dağılımı
    print("\nTest setindeki sınıf dağılımı:")
    unique, counts = np.unique(test_labels, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  {SHORT_CLASS_NAMES[u]}: {c}")
    
    # Değerlendirme
    predictions, probabilities = evaluate_model(model, test_images, test_labels)
    
    # Sonuçları yazdır
    accuracy = print_results(test_labels, predictions)
    
    # Görselleştirmeler
    print("\nGörselleştirmeler oluşturuluyor...")
    plot_confusion_matrix(test_labels, predictions, 'checkpoints_tomato_baseline/confusion_matrix.png')
    plot_per_class_accuracy(test_labels, predictions, 'checkpoints_tomato_baseline/per_class_accuracy.png')
    
    # Özet
    print("\n" + "="*70)
    print("TEST TAMAMLANDI!")
    print("="*70)
    print(f"📊 Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"📁 Sonuçlar: checkpoints_tomato_baseline/")
    print("="*70)


if __name__ == '__main__':
    main()
