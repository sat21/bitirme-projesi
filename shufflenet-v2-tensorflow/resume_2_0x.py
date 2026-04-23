"""
ShuffleNet V2 2.0x Resume Training Script
Eğitimi kaldığı yerden devam ettirir (Epoch 16'dan)
"""

import tensorflow as tf
import numpy as np
import os
import glob
from PIL import Image
import time

# GPU ayarları
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Hiperparametreler
BATCH_SIZE = 16
EPOCHS = 50
INITIAL_EPOCH = 16  # Kaldığı epoch
LEARNING_RATE = 0.01
IMAGE_SIZE = 224
NUM_CLASSES = 10
TRAIN_SPLIT = 0.8

# Veri seti yolu
DATA_DIR = '/mnt/021630F41630E9F5/PROJECTS/torch/tomato'

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
    img = Image.open(image_path).convert('RGB')
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    img = np.array(img, dtype=np.float32)
    img = img / 255.0
    img = (img - 0.5) / 0.5
    return img


def split_data(images, labels, train_ratio=0.8):
    """Eğitim ve doğrulama setlerine ayır"""
    np.random.seed(42)  # Aynı split için sabit seed
    indices = np.random.permutation(len(images))
    images = images[indices]
    labels = labels[indices]
    
    n_train = int(len(images) * train_ratio)
    
    return images[:n_train], labels[:n_train], images[n_train:], labels[n_train:]


def create_dataset(image_paths, labels, batch_size, shuffle=True):
    """TensorFlow Dataset oluştur"""
    
    def load_and_preprocess(img_path, label):
        img = tf.numpy_function(preprocess_image, [img_path], tf.float32)
        img.set_shape([IMAGE_SIZE, IMAGE_SIZE, 3])
        return img, label
    
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_paths))
    
    dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


# ==================== Main ====================

def main():
    print("="*60)
    print("ShuffleNet V2 2.0x - RESUME Training")
    print(f"Starting from Epoch {INITIAL_EPOCH + 1}")
    print("="*60)
    
    # Veri setini yükle
    images, labels = load_dataset()
    print(f"\nToplam görüntü sayısı: {len(images)}")
    
    # Split
    train_images, train_labels, val_images, val_labels = split_data(
        images, labels, TRAIN_SPLIT
    )
    print(f"Eğitim seti: {len(train_images)} görüntü")
    print(f"Doğrulama seti: {len(val_images)} görüntü")
    
    # Dataset
    train_dataset = create_dataset(train_images, train_labels, BATCH_SIZE, shuffle=True)
    val_dataset = create_dataset(val_images, val_labels, BATCH_SIZE, shuffle=False)
    
    # Model yükle
    checkpoint_dir = './checkpoints_tomato_2_0x_baseline'
    model_path = os.path.join(checkpoint_dir, 'best_model.keras')
    
    print(f"\nModel yükleniyor: {model_path}")
    
    custom_objects = {
        'ChannelShuffle': ChannelShuffle,
        'ChannelSplit': ChannelSplit
    }
    
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    print(f"Model yüklendi: {model.count_params():,} parametre")
    
    # Optimizer - mevcut LR'yi epoch'a göre ayarla
    current_lr = LEARNING_RATE * (0.1 ** (INITIAL_EPOCH // 20))
    print(f"Current Learning Rate: {current_lr}")
    
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=current_lr,
        momentum=0.9
    )
    
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, 'best_model.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.LearningRateScheduler(
            lambda epoch: LEARNING_RATE * (0.1 ** (epoch // 20)),
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(checkpoint_dir, 'training_log.csv'),
            append=True  # Mevcut log'a ekle
        )
    ]
    
    # Eğitime devam et
    print("\n" + "="*60)
    print(f"Eğitim devam ediyor... (Epoch {INITIAL_EPOCH + 1} - {EPOCHS})")
    print("="*60 + "\n")
    
    start_time = time.time()
    
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        initial_epoch=INITIAL_EPOCH,
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - start_time
    
    # Sonuçları yazdır
    print("\n" + "="*60)
    print("Eğitim Tamamlandı!")
    print("="*60)
    print(f"Ek eğitim süresi: {training_time/60:.2f} dakika")
    print(f"En iyi doğrulama doğruluğu: {max(history.history['val_accuracy']):.4f}")
    print(f"Son doğrulama doğruluğu: {history.history['val_accuracy'][-1]:.4f}")
    
    # Final modeli kaydet
    model.save(os.path.join(checkpoint_dir, 'final_model.keras'))
    print(f"Final model kaydedildi: {os.path.join(checkpoint_dir, 'final_model.keras')}")


if __name__ == '__main__':
    main()
