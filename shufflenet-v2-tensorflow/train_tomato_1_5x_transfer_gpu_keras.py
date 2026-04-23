# Keras ile ShuffleNetV2 1.5x (transfer learning, data augmentation yok, GPU)
# Not: ShuffleNetV2 Keras implementasyonu gerektirir (pip install keras-shufflenet-v2)

import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
import tensorflow as tf


# Projeye eklenen ShuffleNetV2 Keras implementasyonu
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from shufflenetv2_keras import ShuffleNetV2

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
    print("GPU bulunamadı, CPU kullanılacak.")

BATCH_SIZE = 32
IMAGE_SIZE = 224
NUM_CLASSES = 10
EPOCHS = 50
TRAIN_SPLIT = 0.8
DATA_DIR = '/mnt/021630F41630E9F5/PROJECTS/torch/tomato'
CHECKPOINT_DIR = './checkpoints_tomato_1_5x_transfer_gpu_keras'
MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'best_model.keras')

if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

# Data generator (augmentation yok)
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=1-TRAIN_SPLIT
)

train_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)



# ShuffleNetV2 1.5x backbone (projeye eklenen Keras implementasyonu, son katman hariç)
base_model = ShuffleNetV2(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), n_classes=NUM_CLASSES, scale_factor=1.5)
base_model.trainable = False  # Sadece transfer learning

# Son Dense katmanı hariç tüm katmanları kullan
feature_extractor = Model(inputs=base_model.input, outputs=base_model.layers[-3].output)

inputs = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
x = feature_extractor(inputs)
x = Dense(256, activation='relu')(x)
outputs = Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(inputs, outputs)

model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
earlystop = EarlyStopping(monitor='val_accuracy', patience=50, restore_best_weights=True)

print('Model eğitiliyor (ShuffleNetV2 1.5x transfer learning, augmentation yok, GPU)...')
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=[checkpoint, earlystop]
)

print('Eğitim tamamlandı. En iyi model kaydedildi:', MODEL_PATH)

# Final modeli kaydet
final_model_path = os.path.join(CHECKPOINT_DIR, 'final_model.keras')
model.save(final_model_path)
print('Final model kaydedildi:', final_model_path)

# Eğitim geçmişini kaydet
log_path = os.path.join(CHECKPOINT_DIR, 'training_log.csv')
pd.DataFrame(history.history).to_csv(log_path, index=False)
print('Eğitim logu kaydedildi:', log_path)
