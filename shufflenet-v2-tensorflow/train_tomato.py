import tensorflow as tf
import numpy as np
import os
import glob
from PIL import Image
from net import ShuffleNetV2
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
DATA_DIR = '/home/excalibur/Documents/torch/tomato'

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

def batch_generator(images, labels, batch_size, is_training=True):
    """Batch generator"""
    n_samples = len(images)
    indices = np.arange(n_samples)
    
    if is_training:
        np.random.shuffle(indices)
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        
        batch_images = []
        batch_labels = []
        
        for idx in batch_indices:
            img = preprocess_image(images[idx])
            batch_images.append(img)
            batch_labels.append(labels[idx])
        
        yield np.array(batch_images), np.array(batch_labels)

def main():
    # Veri setini yükle
    images, labels = load_dataset()
    print(f"\nToplam görüntü sayısı: {len(images)}")
    print(f"Sınıf sayısı: {NUM_CLASSES}")
    
    # Eğitim ve doğrulama setlerine ayır
    train_images, train_labels, val_images, val_labels = split_data(
        images, labels, TRAIN_SPLIT
    )
    print(f"Eğitim seti: {len(train_images)} görüntü")
    print(f"Doğrulama seti: {len(val_images)} görüntü")
    
    # TensorFlow graph oluştur
    tf.reset_default_graph()
    
    # Placeholders
    input_placeholder = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, 3], name='input')
    labels_placeholder = tf.placeholder(tf.int64, [None], name='labels')
    is_training = tf.placeholder(tf.bool, name='is_training')
    
    # Model - ShuffleNet V2 1.0 (transfer öğrenme olmadan, sıfırdan)
    model = ShuffleNetV2(input_placeholder, NUM_CLASSES, model_scale=1.0, is_training=is_training)
    logits = model.output
    
    # Loss
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_placeholder, logits=logits)
    )
    
    # Accuracy
    predictions = tf.argmax(logits, axis=1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, labels_placeholder), tf.float32))
    
    # Optimizer
    global_step = tf.Variable(0, trainable=False)
    
    # Learning rate decay
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE,
        global_step,
        decay_steps=1000,
        decay_rate=0.96,
        staircase=True
    )
    
    # Update ops for batch norm
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
        train_op = optimizer.minimize(loss, global_step=global_step)
    
    # Saver
    saver = tf.train.Saver(max_to_keep=5)
    
    # Checkpoint dizini
    checkpoint_dir = './checkpoints_tomato'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # Session config
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    # Eğitim
    print("\n" + "="*60)
    print("ShuffleNet V2 1.0 Baseline Eğitimi Başlıyor")
    print("Transfer Öğrenme: YOK")
    print("Data Augmentation: YOK")
    print("="*60 + "\n")
    
    best_val_acc = 0.0
    
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        
        n_train_batches = (len(train_images) + BATCH_SIZE - 1) // BATCH_SIZE
        n_val_batches = (len(val_images) + BATCH_SIZE - 1) // BATCH_SIZE
        
        for epoch in range(EPOCHS):
            epoch_start = time.time()
            
            # Eğitim
            train_loss = 0.0
            train_acc = 0.0
            n_batches = 0
            
            for batch_images, batch_labels in batch_generator(train_images, train_labels, BATCH_SIZE, True):
                _, batch_loss, batch_acc = sess.run(
                    [train_op, loss, accuracy],
                    feed_dict={
                        input_placeholder: batch_images,
                        labels_placeholder: batch_labels,
                        is_training: True
                    }
                )
                train_loss += batch_loss
                train_acc += batch_acc
                n_batches += 1
                
                if n_batches % 50 == 0:
                    print(f"  Epoch {epoch+1}/{EPOCHS} - Batch {n_batches}/{n_train_batches} - "
                          f"Loss: {batch_loss:.4f} - Acc: {batch_acc:.4f}")
            
            train_loss /= n_batches
            train_acc /= n_batches
            
            # Doğrulama
            val_loss = 0.0
            val_acc = 0.0
            n_batches = 0
            
            for batch_images, batch_labels in batch_generator(val_images, val_labels, BATCH_SIZE, False):
                batch_loss, batch_acc = sess.run(
                    [loss, accuracy],
                    feed_dict={
                        input_placeholder: batch_images,
                        labels_placeholder: batch_labels,
                        is_training: False
                    }
                )
                val_loss += batch_loss
                val_acc += batch_acc
                n_batches += 1
            
            val_loss /= n_batches
            val_acc /= n_batches
            
            epoch_time = time.time() - epoch_start
            
            print(f"\nEpoch {epoch+1}/{EPOCHS} ({epoch_time:.1f}s)")
            print(f"  Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
            
            # En iyi modeli kaydet
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_path = saver.save(sess, os.path.join(checkpoint_dir, 'best_model.ckpt'))
                print(f"  [*] En iyi model kaydedildi: {save_path} (Val Acc: {val_acc:.4f})")
            
            # Her 10 epoch'ta kaydet
            if (epoch + 1) % 10 == 0:
                save_path = saver.save(sess, os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.ckpt'))
                print(f"  Model kaydedildi: {save_path}")
            
            print()
        
        # Son modeli kaydet
        save_path = saver.save(sess, os.path.join(checkpoint_dir, 'final_model.ckpt'))
        print(f"\nEğitim tamamlandı!")
        print(f"Son model kaydedildi: {save_path}")
        print(f"En iyi doğrulama doğruluğu: {best_val_acc:.4f}")

if __name__ == '__main__':
    main()
