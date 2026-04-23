"""
ShuffleNet V2 1.0x Transfer Learning Training Script
Sadece transfer learning, data augmentation yok
"""

import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
import pandas as pd

# GPU ayarları
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

BATCH_SIZE = 32
IMAGE_SIZE = 224
NUM_CLASSES = 10
EPOCHS = 50
TRAIN_SPLIT = 0.8
DATA_DIR = '/mnt/021630F41630E9F5/PROJECTS/torch/tomato'
CHECKPOINT_DIR = './checkpoints_tomato_1_0x_transfer'
MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'best_model.keras')

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

# Data generator (sadece yeniden ölçekleme, augmentation yok)
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

# Model: MobileNetV2 backbone + yeni dense layer
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
base_model.trainable = False  # Sadece transfer learning

inputs = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
outputs = Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(inputs, outputs)

model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
earlystop = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

print('Model eğitiliyor (1.0x transfer learning, augmentation yok)...')
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
