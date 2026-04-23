# Bitirme Projesi

Bu depo; domates hastalığı sınıflandırma, model eğitimi, dağıtım çıktıları ve Android entegrasyonunu içeren bitirme projesinin tam çalışma alanıdır.

## Depo Yapısı

- `shufflenet-v2-tensorflow/`: Eğitim, değerlendirme, dönüştürme ve dağıtım betikleri.
- `tomatech-android/`: Cihaz üzerinde çıkarım yapan Android uygulaması.
- `tomato/`: Eğitim akışlarında kullanılan domates görüntü veri seti.
- `checkpoints_tomato_*`: Farklı deneylere ait kayıtlı checkpoint ve değerlendirme çıktıları.
- `teknofest_2026_on_degerlendirme/`: Raporlama ve inceleme materyalleri.

## Hızlı Başlangıç (Linux)

1. Bir sanal ortam oluşturun ve etkinleştirin.

~~~bash
python3 -m venv .venv
source .venv/bin/activate
~~~

2. Proje bağımlılıklarını kurun.

~~~bash
pip install --upgrade pip
pip install -r requirements.txt
~~~

3. İlgili alt klasörden eğitim veya değerlendirme betiklerini çalıştırın.

~~~bash
cd shufflenet-v2-tensorflow
python train_tomato_1_5x_aug.py
~~~

## Git İş Akışı

- Ana dal: `main`
- Riskli değişiklikler için özellik dalları kullanın.
- Küçük commitler atın ve sık push yapın.
- Ortak geçmişi düzeltirken yıkıcı yeniden yazımlar yerine `git revert` kullanın.

## Büyük Dosya Politikası

Bu depo, yeniden üretilebilirlik için gerekli önemli model çıktıları ve veri seti varlıklarını bilinçli olarak takip eder.

`.gitignore` dosyası; yerel ortamları, taşıma yedeklerini ve gelecekte üretilecek çalışma zamanı/günlük/önbellek çıktılarının yeni commitlere karışmasını engeller.

## Mevcut Uzak Repo

- GitHub: https://github.com/sat21/bitirme-projesi
