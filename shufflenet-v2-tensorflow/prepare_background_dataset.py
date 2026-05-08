import os
import urllib.request
import time

def prepare_background_class(data_dir):
    """
    Bu script, internetten modelin 'domates olmayan' resimleri taniyabilmesi
    icin rastgele 'Background_Out_Of_Domain' (OOD) sinifini doldurmaya yardimci olur.
    """
    bg_dir = os.path.join(data_dir, "Background_Out_Of_Domain")
    os.makedirs(bg_dir, exist_ok=True)
    
    # Eger klasorde hali hazirda resim varsa hicbir sey yapma
    existing_files = [f for f in os.listdir(bg_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if len(existing_files) > 100:
        print(f"[BILGI] {bg_dir} zaten yeterli arka plan resmiyle dolu.")
        return

    print(f"[BILGI] {bg_dir} klasoru hazirlaniyor...")
    print("Bu islem rastgele genel gorseller indirecektir (Manzara, Hayvan, Esya vb.)")
    print("Indiriliyor... Lutfen bekleyin.")
    
    # 200 adet rastgele resim indirelim (Lorem Picsum uzerinden)
    target_count = 200
    current_count = len(existing_files)
    
    for i in range(current_count, target_count):
        try:
            # Her seferinde farkli bir resim gelmesi icin random seed (v=) kullaniyoruz
            url = f"https://picsum.photos/400/400?random={i}"
            save_path = os.path.join(bg_dir, f"bg_{i:04d}.jpg")
            urllib.request.urlretrieve(url, save_path)
            
            if (i+1) % 20 == 0:
                print(f"  -> {i+1}/{target_count} resim indirildi.")
                
            time.sleep(0.1)  # Sunucuyu yormamak icin ufacik bir bekleme
        except Exception as e:
            print(f"Indirme hatasi: {e}")

if __name__ == "__main__":
    # Egitim scriptindeki klasor dizinini kullaniyoruz
    DATA_DIR = '/mnt/50267C3D267C265E/yeni birim/PROJECTS/torch/tomato'
    prepare_background_class(DATA_DIR)
    
    print("\n--------------------------------------------------------------")
    print("TAMAMLANDI!")
    print("Artik modeliniz domates disi her seyi 'Background_Out_Of_Domain' olarak ogrenecek.")
    print("--------------------------------------------------------------")
