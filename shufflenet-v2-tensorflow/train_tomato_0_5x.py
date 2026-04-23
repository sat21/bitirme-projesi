"""
ShuffleNet V2 0.5x Baseline Training for Tomato Dataset
TensorFlow 2.x / Keras Implementation

- Transfer öğrenme YOK (sıfırdan eğitim)
- Data augmentation YOK
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
BATCH_SIZE = 32
EPOCHS = 50
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


# ==================== ShuffleNet V2 Model ====================

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
        # --- Data Augmentation ---
        import random

        channels_per_group = channels // self.groups
        
        x = tf.reshape(x, [batch_size, height, width, self.groups, channels_per_group])
        x = tf.transpose(x, [0, 1, 2, 4, 3])
        x = tf.reshape(x, [batch_size, height, width, channels])
        
        return x
    

        def preprocess_image_train(image_path):
            """Eğitim için: resize, normalize ve data augmentation uygula"""
            img = Image.open(image_path).convert('RGB')
            # Random horizontal flip
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            # Random rotation
            if random.random() < 0.5:
                angle = random.uniform(-20, 20)
                img = img.rotate(angle)
            # Random brightness
            if random.random() < 0.5:
                factor = random.uniform(0.8, 1.2)
                img = Image.fromarray(np.clip(np.array(img) * factor, 0, 255).astype(np.uint8))
            # Resize
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
            img = np.array(img, dtype=np.float32)
            img = img / 255.0
            img = (img - 0.5) / 0.5
            return img

        def preprocess_image_val(image_path):
            """Doğrulama için: sadece resize ve normalize"""
            img = Image.open(image_path).convert('RGB')
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
            img = np.array(img, dtype=np.float32)
            img = img / 255.0
            img = (img - 0.5) / 0.5
            return img

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
def conv_bn_relu(x, out_channels, kernel_size, strides=1):
    """Conv + BatchNorm + ReLU"""
    x = tf.keras.layers.Conv2D(
        out_channels, kernel_size, strides=strides, padding='same',
        use_bias=False,
        kernel_initializer='he_normal'
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x


def depthwise_conv_bn(x, kernel_size, strides=1):
    """Depthwise Conv + BatchNorm"""
    x = tf.keras.layers.DepthwiseConv2D(
        kernel_size, strides=strides, padding='same',
        use_bias=False,
        depthwise_initializer='he_normal'
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return x


def shufflenet_v2_block(x, out_channels, strides=1, shuffle_group=2):
    """ShuffleNet V2 basic block"""
    
    if strides == 1:
        # Channel split using custom layer
        top = ChannelSplit(0)(x)
        bottom = ChannelSplit(1)(x)
        
        # Right branch
        half_channels = out_channels // 2
        top = conv_bn_relu(top, half_channels, 1)
        top = depthwise_conv_bn(top, 3, strides)
        top = conv_bn_relu(top, half_channels, 1)
        
        # Concat
        out = tf.keras.layers.Concatenate(axis=-1)([top, bottom])
        
        # Channel shuffle
        out = ChannelShuffle(groups=shuffle_group)(out)
        
    else:
        # Stride = 2, downsample
        half_channels = out_channels // 2
        
        # Left branch
        left = depthwise_conv_bn(x, 3, strides)
        left = conv_bn_relu(left, half_channels, 1)
        
        # Right branch
        right = conv_bn_relu(x, half_channels, 1)
        right = depthwise_conv_bn(right, 3, strides)
        right = conv_bn_relu(right, half_channels, 1)
        
        # Concat
        out = tf.keras.layers.Concatenate(axis=-1)([left, right])
        
        # Channel shuffle
        out = ChannelShuffle(groups=shuffle_group)(out)
    
    return out


def build_shufflenet_v2(input_shape=(224, 224, 3), num_classes=10, model_scale=0.5):
    """Build ShuffleNet V2 model from scratch (no pretrained weights)"""
    
    # Channel configuration based on model scale
    if model_scale == 0.5:
        stage_out_channels = [24, 48, 96, 192, 1024]
    elif model_scale == 1.0:
        stage_out_channels = [24, 116, 232, 464, 1024]
    elif model_scale == 1.5:
        stage_out_channels = [24, 176, 352, 704, 1024]
    elif model_scale == 2.0:
        stage_out_channels = [24, 244, 488, 976, 2048]
    else:
        raise ValueError(f"Unsupported model_scale: {model_scale}")
    
    stage_repeats = [4, 8, 4]
    
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # Stage 1: Conv + MaxPool
    x = tf.keras.layers.Conv2D(
        stage_out_channels[0], 3, strides=2, padding='same',
        use_bias=False,
        kernel_initializer='he_normal'
    )(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)
    
    # Stage 2, 3, 4
    for stage_idx in range(3):
        out_channels = stage_out_channels[stage_idx + 1]
        repeat = stage_repeats[stage_idx]
        
        # First block with stride=2
        x = shufflenet_v2_block(x, out_channels, strides=2)
        
        # Remaining blocks with stride=1
        for _ in range(repeat - 1):
            x = shufflenet_v2_block(x, out_channels, strides=1)
    
    # Stage 5: Conv
    x = conv_bn_relu(x, stage_out_channels[-1], 1)
    
    # Global Average Pooling
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    # Classifier
    outputs = tf.keras.layers.Dense(num_classes, kernel_initializer='he_normal')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='ShuffleNetV2_0.5')
    
    return model


# ==================== Data Loading ====================

def load_dataset():
    """Veri setini yükle - data augmentation olmadan"""
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
    """Görüntüyü ön işle - sadece resize ve normalize (augmentation yok)"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    img = np.array(img, dtype=np.float32)
    img = img / 255.0  # Normalize [0, 1]
    img = (img - 0.5) / 0.5  # Normalize [-1, 1]
    return img


def shuffle_data(images, labels):
    """Veriyi karıştır"""
    np.random.seed(42)
    indices = np.random.permutation(len(images))
    return images[indices], labels[indices]


def split_data(images, labels, train_ratio=0.8):
    """Eğitim ve doğrulama setlerine ayır"""
    n_samples = len(images)
    n_train = int(n_samples * train_ratio)
    
    # Önce karıştır
    images, labels = shuffle_data(images, labels)
    
    train_images = images[:n_train]
    train_labels = labels[:n_train]
    val_images = images[n_train:]
    val_labels = labels[n_train:]
    
    return train_images, train_labels, val_images, val_labels


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


# ==================== Training ====================

def main():
    print("="*60)
    print("ShuffleNet V2 0.5x Baseline Training")
    print("Transfer Learning: DISABLED")
    print("Data Augmentation: DISABLED")
    print("="*60)
    
    # Veri setini yükle
    images, labels = load_dataset()
    print(f"\nToplam görüntü sayısı: {len(images)}")
    print(f"Sınıf sayısı: {NUM_CLASSES}")
    
    # Eğitim ve doğrulama setlerine ayır
    train_images, train_labels, val_images, val_labels = split_data(
        images, labels, TRAIN_SPLIT
    )
    print(f"\nEğitim seti: {len(train_images)} görüntü")
    print(f"Doğrulama seti: {len(val_images)} görüntü")
    
    # Dataset oluştur
    train_dataset = create_dataset(train_images, train_labels, BATCH_SIZE, shuffle=True)
    val_dataset = create_dataset(val_images, val_labels, BATCH_SIZE, shuffle=False)
    
    # Model oluştur (sıfırdan, pretrained weights yok)
    print("\nModel oluşturuluyor (ShuffleNet V2 0.5x - sıfırdan, transfer öğrenme yok)...")
    model = build_shufflenet_v2(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        num_classes=NUM_CLASSES,
        model_scale=0.5
    )
    
    model.summary()
    
    # Optimizer ve loss
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=LEARNING_RATE,
        momentum=0.9
    )
    
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    # Compile
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=['accuracy']
    )
    
    # Callbacks
    checkpoint_dir = './checkpoints_tomato_0_5x_baseline'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
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
            os.path.join(checkpoint_dir, 'training_log.csv')
        )
    ]
    
    # Eğitim
    print("\n" + "="*60)
    print("Eğitim başlıyor...")
    print("="*60 + "\n")
    
    start_time = time.time()
    
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - start_time
    
    # Sonuçları yazdır
    print("\n" + "="*60)
    print("Eğitim Tamamlandı!")
    print("="*60)
    print(f"Toplam eğitim süresi: {training_time/60:.2f} dakika")
    print(f"En iyi doğrulama doğruluğu: {max(history.history['val_accuracy']):.4f}")
    print(f"Son doğrulama doğruluğu: {history.history['val_accuracy'][-1]:.4f}")
    print(f"Model kaydedildi: {checkpoint_dir}")
    
    # Final modeli kaydet
    model.save(os.path.join(checkpoint_dir, 'final_model.keras'))
    print(f"Final model kaydedildi: {os.path.join(checkpoint_dir, 'final_model.keras')}")


if __name__ == '__main__':
    main()
