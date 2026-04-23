import tensorflow as tf
import os
from shufflenetv2_keras import ShuffleNetV2

# Eski model dosyası (Lambda ile kaydedilmiş)
OLD_MODEL_PATH = './checkpoints_tomato_1_5x_transfer_gpu_noaug/best_model.keras'
# Yeni model dosyası (ChannelShuffleLayer ile kaydedilecek)
NEW_MODEL_PATH = './checkpoints_tomato_1_5x_transfer_gpu_noaug/best_model_layer.keras'

# Model parametreleri
def get_model():
    return ShuffleNetV2(input_shape=(224,224,3), n_classes=10, scale_factor=1.5)

# 1. Yeni mimariyi oluştur
model = get_model()

# 2. Eski modelin ağırlıklarını yükle (Lambda closure hatası nedeniyle doğrudan load_model kullanılamaz)
try:
    old_model = tf.keras.models.load_model(OLD_MODEL_PATH, compile=False, safe_mode=False)
    print('Eski model başarıyla yüklendi.')
    # 3. Ağırlıkları yeni modele aktar
    model.set_weights(old_model.get_weights())
    print('Ağırlıklar yeni modele aktarıldı.')
    # 4. Yeni modeli kaydet
    model.save(NEW_MODEL_PATH)
    print('Yeni model kaydedildi:', NEW_MODEL_PATH)
except Exception as e:
    print('Ağırlık aktarımı başarısız:', e)
