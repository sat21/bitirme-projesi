"""
ShuffleNet V2 1.5x Transfer Learning Training Script (GPU, 50 epoch)
Sadece transfer learning, data augmentation yok
"""

import tensorflow as tf
import os
from net import ShuffleNetV2
import numpy as np
from tensorflow.contrib import slim
from tensorflow.python.framework import ops
from sklearn.model_selection import train_test_split
from glob import glob
from PIL import Image

# GPU ayarları
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
CHECKPOINT_DIR = './checkpoints_tomato_1_5x_transfer_gpu'
MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'best_model.ckpt')
FINAL_MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'final_model.ckpt')
LOG_PATH = os.path.join(CHECKPOINT_DIR, 'training_log.csv')

if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

# Veri yükleme fonksiyonları (Kendi loader'ınız varsa ekleyin)
def load_data(data_dir, image_size, train_split):
    class_names = sorted(os.listdir(data_dir))
    X, y = [], []
    for idx, cname in enumerate(class_names):
        img_paths = glob(os.path.join(data_dir, cname, '*.jpg'))
        for img_path in img_paths:
            img = Image.open(img_path).convert('RGB').resize((image_size, image_size))
            X.append(np.array(img) / 255.0)
            y.append(idx)
    X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)
    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=train_split, stratify=y, random_state=42)
    return X_train, y_train, X_val, y_val

ops.reset_default_graph()
input_holder = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, 3], name='input')
label_holder = tf.placeholder(tf.int32, [None], name='label')

model = ShuffleNetV2(input_holder, NUM_CLASSES, model_scale=1.5, is_training=True)
logits = model.output

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_holder, logits=logits))
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)
pred = tf.argmax(logits, axis=1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, label_holder), tf.float32))

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    X_train, y_train, X_val, y_val = load_data(DATA_DIR, IMAGE_SIZE, TRAIN_SPLIT)
    best_val_acc = 0
    log = []
    for epoch in range(EPOCHS):
        # Mini-batch training
        idxs = np.arange(len(X_train))
        np.random.shuffle(idxs)
        for i in range(0, len(X_train), BATCH_SIZE):
            batch_idx = idxs[i:i+BATCH_SIZE]
            batch_x = X_train[batch_idx]
            batch_y = y_train[batch_idx]
            sess.run(optimizer, feed_dict={input_holder: batch_x, label_holder: batch_y})
        # Validation
        val_acc, val_loss = sess.run([accuracy, loss], feed_dict={input_holder: X_val, label_holder: y_val})
        log.append({'epoch': epoch, 'val_accuracy': val_acc, 'val_loss': val_loss})
        print(f"Epoch {epoch+1}/{EPOCHS} - val_acc: {val_acc:.4f} val_loss: {val_loss:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            saver.save(sess, MODEL_PATH)
    saver.save(sess, FINAL_MODEL_PATH)
    print('Final model kaydedildi:', FINAL_MODEL_PATH)
    import pandas as pd
    pd.DataFrame(log).to_csv(LOG_PATH, index=False)
    print('Eğitim logu kaydedildi:', LOG_PATH)
