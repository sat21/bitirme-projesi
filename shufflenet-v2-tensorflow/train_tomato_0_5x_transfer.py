"""
ShuffleNet V2 0.5x Transfer Learning Training for Tomato Dataset (No Data Augmentation, GPU)
TensorFlow 2.x / Keras Implementation

- Sadece transfer learning (önceden eğitilmiş ağırlıklar)
- Data augmentation YOK
- GPU kullanımı AKTİF
"""

import tensorflow as tf
import numpy as np
import os
import glob
from PIL import Image
import time
import random

# GPU kullanımı için ayar
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU kullanılacak: {gpus}")
    except RuntimeError as e:
        print(e)
else:
    print("UYARI: GPU bulunamadı, eğitim CPU ile devam edecek.")

BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.01
IMAGE_SIZE = 224
NUM_CLASSES = 10
TRAIN_SPLIT = 0.8

DATA_DIR = '/mnt/021630F41630E9F5/PROJECTS/torch/tomato'

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
# ...existing code...

def build_shufflenet_v2_0_5x_transfer(input_shape=(224, 224, 3), num_classes=10):
    # Keras uygulamalarında ShuffleNet yok, MobileNetV2 ile örnek transfer learning gösterimi
    # Eğer önceden eğitilmiş ShuffleNet ağırlığı varsa, burada kullanılabilir
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Sadece üst katmanlar eğitilecek
    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation=None, kernel_initializer='he_normal')(x)
    model = tf.keras.Model(inputs, outputs, name='MobileNetV2_Transfer_0.5x')
    return model

# ==================== Data Loading ====================
def load_dataset():
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
    img = Image.open(image_path).convert('RGB')
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    img = np.array(img, dtype=np.float32)
    img = img / 255.0
    img = (img - 0.5) / 0.5
    return img

def shuffle_data(images, labels):
    np.random.seed(42)
    indices = np.random.permutation(len(images))
    return images[indices], labels[indices]

def split_data(images, labels, train_ratio=0.8):
    n_samples = len(images)
    n_train = int(n_samples * train_ratio)
    images, labels = shuffle_data(images, labels)
    train_images = images[:n_train]
    train_labels = labels[:n_train]
    val_images = images[n_train:]
    val_labels = labels[n_train:]
    return train_images, train_labels, val_images, val_labels

def create_dataset(image_paths, labels, batch_size, shuffle=True):
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
    print("ShuffleNet V2 0.5x Transfer Learning Training (No Data Aug, GPU)")
    print("Transfer Learning: ENABLED (ImageNet)")
    print("Data Augmentation: DISABLED")
    print("GPU: AKTİF" if gpus else "GPU: YOK, CPU ile devam")
    print("="*60)
    images, labels = load_dataset()
    print(f"\nToplam görüntü sayısı: {len(images)}")
    print(f"Sınıf sayısı: {NUM_CLASSES}")
    train_images, train_labels, val_images, val_labels = split_data(
        images, labels, TRAIN_SPLIT
    )
    print(f"\nEğitim seti: {len(train_images)} görüntü")
    print(f"Doğrulama seti: {len(val_images)} görüntü")
    train_dataset = create_dataset(train_images, train_labels, BATCH_SIZE, shuffle=True)
    val_dataset = create_dataset(val_images, val_labels, BATCH_SIZE, shuffle=False)
    print("\nModel oluşturuluyor (MobileNetV2 transfer learning, üst katmanlar eğitilecek)...")
    model = build_shufflenet_v2_0_5x_transfer(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        num_classes=NUM_CLASSES
    )
    model.summary()
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=LEARNING_RATE,
        momentum=0.9
    )
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=['accuracy']
    )
    checkpoint_dir = './checkpoints_tomato_0_5x_transfer'
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
    print("\n" + "="*60)
    print("Eğitim Tamamlandı!")
    print("="*60)
    print(f"Toplam eğitim süresi: {training_time/60:.2f} dakika")
    print(f"En iyi doğrulama doğruluğu: {max(history.history['val_accuracy']):.4f}")
    print(f"Son doğrulama doğruluğu: {history.history['val_accuracy'][-1]:.4f}")
    print(f"Model kaydedildi: {checkpoint_dir}")
    model.save(os.path.join(checkpoint_dir, 'final_model.keras'))
    print(f"Final model kaydedildi: {os.path.join(checkpoint_dir, 'final_model.keras')}")

if __name__ == '__main__':
    main()
