# Otonom Otobüslerde GNSS Bozulmasına Karşı Adaptif EKF Tabanlı Sensör Füzyonu ile Lokalizasyon İyileştirmesi

Bu depo, şehir içi hatlarda çalışan otonom bir elektrikli otobüsün GNSS bozulması, tünel içi sinyal kaybı ve tekerlek kayması (slip) gibi zorlayıcı ortam koşullarında konum doğruluğunu artırmak amacıyla geliştirilen **Adaptif Genişletilmiş Kalman Filtresi (AEKF)** tabanlı lokalizasyon sistemini içermektedir.

Karşılaştırma yöntemi olarak Klasik Genişletilmiş Kalman Filtresi (EKF) ve Mahalanobis kapılama (gating) destekli Unscented Kalman Filtresi (UKF) entegre edilmiştir.

---

## 1. Projenin Amacı ve Özeti
Şehir içi yollarda (urban canyon, tünel, alt geçitler) GNSS alıcılarının sinyal hassasiyeti ciddi oranda düşmekte ve metre mertebesinde konum hataları oluşmaktadır. Bu projede:
*   IMU (İvme, Açısal Hız), Tekerlek Odometrisi (Hız), GNSS (2B Konum) ve LiDAR/Nirengi (Mesafe/Kerteriz) verileri asenkron olarak füzyona dahil edilir.
*   **Adaptif Kovaryans Güncellemesi ($R_k$):** AEKF altyapısı, GNSS alıcısı bozulmalarını yenilik (innovation) residual normu ile izler. Odometri kaymalarını ise predicted hız farklarıyla algılayarak sensör hata kovaryans değerlerini dinamik olarak büyütür veya küçültür.
*   **UKF Gating:** UKF'ye $d^2 = \mathbf{y}^T \mathbf{P}_{zz}^{-1} \mathbf{y} > 9.21$ (\%99 güven sınırı) Mahalanobis kapılaması entegre edilerek, GNSS kanyon yansımaları (outliers) filtreden dışlanır.

---

## 2. Referans Otobüs Modeli
Simülasyonda kullanılan kinematik otobüs boyutları ve manevra sınırları, **AKIA Ultra LF12 EV** elektrikli otobüsünün teknik parametrelerine dayanmaktadır:
*   **Araç Tipi:** 12 m sınıfı alçak tabanlı elektrikli şehir içi otobüs
*   **Uzunluk / Genişlik / Dingil Mesafesi (Wheelbase):** 12,270 mm / 2,540 mm / 6,000 mm
*   **Maksimum Kapasite:** 80 Yolcu (26 Oturan, 54 Ayakta) - Yolcu kütlesi odometri kayma riskini ve gürültü seviyelerini etkiler.
*   **Batarya / Güç:** 465 kWh LFP Batarya / 240 kW Sabit Mıknatıslı Senkron Motor (PMSM)
*   **Fren Sistemi:** EBS, ABS, ASR, ESC ve rejeneratif frenleme (frenleme anlarında IMU ivme ve odometri gürültüsü artırılır).
*   **Gelişmiş Sensör Donanımları:**
    *   *Yol Çizgisi İzleme Kamerası:* Otobüsün çift şeritli yolda kendi şeridinde (sağ şeritte) kalmasını denetleyen yanal sapma takip kamerası ($\sigma_{lane} = 0.02$ m).
    *   *Yakınlık (Ultrasonik) Sensörü:* Durak platformuna yanaşırken çalışan platform mesafe ölçer ($\sigma_{prox} = 0.03$ m).
    *   *Çevre ve Kapı Kameraları:* Kapılarda yolcu biniş/iniş hareketlerini algılayan ve kapı bölgesinde yolcu kalmadığında kapıları otomatik kapatıp kalkış onayını veren sistem.

---

## 3. Kurulum ve Çalıştırma

Gerekli Python paketlerini kurun:
```powershell
python -m pip install -r requirements.txt
```

Projede yer alan tüm analiz, grafik ve animasyon çıktılarını üretmek için komut satırı arayüzü (CLI) parametreleri aşağıda listelenmiştir:

### A) Tekli Simülasyon Koşusu (Nominal Run)
Seed: 42 ve 'combined' senaryo altında tek bir simülasyon çalıştırır, metrikleri ekrana basar ve sonuçları CSV olarak kaydeder:
```powershell
python project/main.py --run-single
```
*   **Çıktı:** `project/outputs/results/results_single_run.csv`

### B) 30 Seed Monte Carlo Simülasyonu
30 farklı random seed için simülasyonu koşturur, IQR tabanlı aykırı değer (outlier) ve ıraksama (divergence) analizlerini gerçekleştirir:
```powershell
python project/main.py --run-monte-carlo --seeds 30
```
*   **Çıktılar:**
    *   `project/outputs/results/results_monte_carlo.csv` (ham seed RMSE verileri)
    *   `project/outputs/results/monte_carlo_summary.csv` (medyan, IQR, divergence ve outlier adetleri)
    *   `project/outputs/results/monte_carlo_outliers.csv` (aykırı veya diverje olan seed listesi)
    *   `project/outputs/figures/outlier_seeds/error_time_seed_*.png` (ıraksayan EKF koşularının hata-zaman grafikleri)

### C) Akademik Grafiklerin Üretilmesi
Akademik raporda kullanılmak üzere yüksek çözünürlüklü (300 DPI) statik grafikleri ilgili subfolder'lara yazar:
```powershell
python project/main.py --generate-figures
```
*   **Çıktılar:** `project/outputs/figures/main/` ve `project/outputs/figures/appendix/` dizinleri.

### D) Canlı Görünüm Statik Karelerinin Üretilmesi
GIF animasyonlarını temsil eden yüksek çözünürlüklü canlı görünüm statik resim karelerini çıkarır:
```powershell
python project/main.py --generate-report-assets
```
*   **Çıktılar:** `project/outputs/report_assets/` dizini.

### E) Simülasyon Animasyonlarının (GIF) Üretilmesi
Matplotlib FuncAnimation tabanlı 5 adet GIF animasyon dosyasını (Nominal Yörünge, Kanyon Geçişi, Durak Yanaşma/Dwell, Kovaryans Değişimi ve Otobüs Takip) oluşturur:
```powershell
python project/main.py --generate-animations
```
*   **Çıktılar:** `project/outputs/animations/` dizini.

### F) RL Durak Yaklaşma Modülü
Pekiştirmeli öğrenme DQN ajanını eğitir ve değerlendirir:
```powershell
python project/main.py --run-rl-stop
```
Alternatif olarak bağımsız eğitim veya değerlendirme komutları çalıştırılabilir:
```powershell
python project/rl_stop/train_dqn_stop.py
```
```powershell
python project/rl_stop/evaluate_stop_policy.py
```
*   **Çıktılar:** `project/outputs/figures/rl_*.png`, `project/outputs/results/rl_stop_results.csv` ve `project/outputs/results/stop_dwell_results.csv`.

---

## 4. Deneysel Bulgular ve Performans Sonuçları

Farklı senaryolar altında elde edilen nominal tekli simülasyon (Seed: 42) ve 30 seed Monte Carlo test sonuçlarının özeti aşağıda sunulmuştur:

*   **Genel Rota Konum RMSE İyileşmesi:** Şerit takip kamerasının füzyona eklenmesiyle nominal rota genelinde Klasik EKF'nin konum RMSE değeri **0.1096 m**'ye, önerilen Adaptif EKF ile **0.0999 m**'ye düşürülerek **%8.89** oranında yüksek bir hassasiyet elde edilmiştir. UKF ise **0.1042 m** hata üretmiştir.
*   **Yol Takip Emniyeti:** Otobüs çift şeritli geliş-gidiş yolda şeridinde (sağ şeritte) kalarak yanal sapmasını sürekli 2.0 cm altında tutmuştur.
*   **Durak Yanaşma Hassasiyeti (s: 200 m):** Durak bölgesinde yanal sapmalar şerit izleme kamerası sayesinde 2.0 cm seviyesinde kalırken, EKF durak boylamsal konum RMSE değeri **0.1047 m**, AEKF ise **0.1334 m** seviyesinde gerçekleşmiştir.
*   **Monte Carlo ve Kararlılık Analizi:** 30 seed Monte Carlo testinde sabit kovaryanslı Klasik EKF, yüksek gürültülü ve tekerlek kaymalı durumların birleştiği 3 tohumda (Seed 5, 12, 24) "Divergence" olarak işaretlenmiştir. AEKF ve Mahalanobis gating destekli UKF ise tüm seed'lerde ıraksamayı tamamen engelleyerek kararlı sınırda kalmışlardır.

---

## 5. RL Stop Control Extension (RL Durak Yaklaşma Modülü)
Bu ek modül, Adaptif EKF ile elde edilen konum kestiriminin kontrol katmanında (durma doğruluğu ve dwell kararı) nasıl kullanılabileceğini gösteren sınırlı kapsamlı bir pekiştirmeli öğrenme ön doğrulamasıdır.

### Modülün Amacı
DQN (Deep Q-Network) tabanlı bir ajan, otobüsün durağa kalan mesafe ve hız bilgisine göre frenleme kararını öğrenir. Ajan, durağa yaklaşıp belirlenen toleranslar ($\pm 0.5$ m) ve hız limitleri ($\le 0.2$ m/s) altında durarak 15 saniyelik yolcu alma/bindirme bekleme süresini (dwell) başarıyla tamamlamayı öğrenir.

### Dosya Yapısı (`project/rl_stop/`)
*   `stop_env.py`: Durak yaklaşma durum uzayını (size: 6, normalize `dwell_ratio` dahil), ayrık aksiyon ivmelerini, 1D kinematik model dinamiklerini ve dwell ödül fonksiyonunu içeren Gym benzeri RL simülasyon ortamıdır.
*   `train_dqn_stop.py`: DQN ajanını PyTorch kullanarak 800 bölüm (episode) boyunca eğiten ve eğitim eğrisini (`rl_stop_training_reward.png`) üreten eğitim kodudur.
*   `evaluate_stop_policy.py`: Eğitilen ajanı kural tabanlı bir durak dwell kontrolcüsü ile karşılaştırmalı olarak test eden, hız-mesafe-dwell profillerini üreten değerlendirme kodudur.
*   `rl_utils.py`: Q-Ağı yapısını (Multi-Layer Perceptron) ve Deneyim Belleğini (Replay Buffer) içeren yardımcı modüldür.

### Üretilen Grafikler ve Sonuçlar
*   `outputs/figures/rl_stop_training_reward.png`: Eğitim boyunca kazanılan toplam ödülün değişim eğrisi. Ajan 800 bölüm sonunda %100 durma ve bekleme başarı oranına ulaşmıştır.
*   `outputs/figures/rl_stop_distance_velocity.png`: DQN ajanının nominal sürüş esnasında zaman serisine göre mesafe, hız ve ivme/dwell komut profili (Seed: 42).
*   `outputs/figures/rl_stop_dwell_behavior.png`: DQN ajanının durma ve bekleme (dwell) anındaki mesafe, hız ve dwell sayaç durum değişim grafikleridir.
*   `outputs/figures/rl_vs_rule_based_stop_error.png`: RL ve kural tabanlı yöntemin ortalama/maksimum durma hatası ve başarı oranı karşılaştırması. DQN ortalama durma hatasını **0.25 m** seviyesine çekerken, kural tabanlı kontrol **7.25 m** hata üretmiştir (yaklaşık **%96.6** hata azalımı).
*   `outputs/results/rl_stop_results.csv` ve `stop_dwell_results.csv`: Ortalama yavaşlama, konfor, dwell başarı ve durma hatası sayısal sonuç tablosu.

### Sınırlamalar ve Güvenlik Uyarıları
*   DQN ajanı basitleştirilmiş 1B kinematik model üzerinde eğitilmiştir. Gerçek araç gövde dinamikleri, yol eğimi, lastik kuvvetleri, yolcu konforu, trafik etkileşimi ve düşük seviyeli fren kontrolü ayrıntılı modellenmemiştir.
*   İlk kontrol tasarımlarında karşılaşılan durağı geçme (overshot) riski, pekiştirmeli öğrenme tabanlı otonom sürüş sistemlerinde ek emniyet filtrelerinin, kısıtlanmış aksiyon uzaylarının veya güvenlik bariyerlerinin entegrasyon gereksinimini göstermektedir. Bu nedenle sonuçlar gerçek araç kontrol başarımı olarak değil, yöntemsel bir ön doğrulama çalışması olarak yorumlanmalıdır.

---

## 6. Çıktı Dosya ve Grafik Sınıflandırması

Tüm çıktılar `project/outputs/` altında organize edilmiştir:

```
project/outputs/
│
├── figures/
│   ├── main/
│   │   ├── trajectory_comparison.png (Yörünge kestirimi genel)
│   │   ├── trajectory_comparison_zoom_a.png (GNSS Bozulması - EKF & AEKF)
│   │   ├── trajectory_comparison_zoom_b.png (GNSS Bozulması - UKF ile karşılaştırma)
│   │   ├── trajectory_comparison_stop.png (Durak 1 yanaşma zoom ve Bilgi Kutusu)
│   │   ├── stop_error_comparison_bar.png (Durak yanal/boylamsal RMSE bar grafiği)
│   │   ├── rmse_comparison_nominal.png (Nominal RMSE bar grafiği)
│   │   ├── position_error_time_full.png (Zamana göre konum hatası tam ölçek)
│   │   ├── position_error_time_zoomed_ekf_aekf.png (Hata yakınlaştırılmış - EKF & AEKF)
│   │   ├── covariance_traces.png (R_gps, R_odom, R_lidar zaman izleri)
│   │   ├── covariance_adaptation_error.png (Kovaryans adaptasyonu hata korelasyonu)
│   │   └── rmse_boxplot_log.png (Monte Carlo kutu grafiği - Logaritmik ölçek)
│   │
│   ├── appendix/
│   │   ├── position_error_time_zoomed_all.png (Zoomed EKF, AEKF, UKF)
│   │   └── rmse_boxplot_linear.png (Monte Carlo kutu grafiği - Doğrusal ölçek)
│   │
│   ├── outlier_seeds/
│   │   └── error_time_seed_*.png (EKF'nin ıraksadığı tohumların konum hata serileri)
│   │
│   ├── rl_stop_training_reward.png (DQN eğitim ödülleri)
│   ├── rl_stop_distance_velocity.png (RL yaklaşma hız-mesafe profili)
│   ├── rl_stop_dwell_behavior.png (RL bekleme sayaç ve hız grafik serisi)
│   └── rl_vs_rule_based_stop_error.png (RL ve kural tabanlı karşılaştırma bar grafiği)
│
├── animations/
│   ├── bus_trajectory_with_stops.gif (Çift şeritli yörünge takip, duraklar ve yolcular)
│   ├── gnss_degradation_zoom.gif (Kanyon geçişi anlık hata sönümleme)
│   ├── bus_stop_dwell.gif (Dwell beklemesi, door camera, binen/inen yolcu dinamikleri)
│   ├── covariance_adaptation.gif (Dinamik gürültü matrisi adaptasyonu)
│   └── bus_focus_tracking.gif (Kameranın otobüsü takip ettiği yakın görünüm - fps: 5)
│
├── results/
│   ├── results_single_run.csv
│   ├── results_monte_carlo.csv
│   ├── monte_carlo_summary.csv
│   ├── monte_carlo_outliers.csv
│   ├── rl_stop_results.csv (DQN ve kural tabanlı durma karşılaştırma tablosu)
│   └── stop_dwell_results.csv (Dwell bekleme detaylı metrikleri)
│
└── report_assets/
    ├── trajectory_frame.png
    ├── degradation_frame.png
    ├── stop_frame.png
    └── covariance_frame.png
```

---

## 7. Simülasyon Varsayımları ve Sınırlamalar
Lokalizasyon başarısının akademik tutarlılığı için simülasyon kısıtlamaları göz önünde bulundurulmalıdır:
1.  **2B Kinematik Varsayımı:** Otobüsün hareketi 2B düzlemde Ackermann direksiyon modeliyle temsil edilmiştir. Yol eğimi, lastik dinamiği ve süspansiyon etkileri ihmal edilmiştir.
2.  **Sentetik Gürültü:** Ölçüm hataları ve gürültüleri sentetik Gaussian dağılımı ve yapay kaymalar eklenerek üretilmiştir. Yol çizgisi takip kamerası 2 cm yanal çözünürlükle modellenmiştir.
3.  **Soyut LiDAR Nirengileri:** Gerçek LiDAR nokta bulutu işleme (ICP, SLAM) yerine, landmark koordinatlarını doğrudan veren mesafe-açı sensör modeli kullanılmıştır.
4.  **Hata Kapılama Gereksinimi:** Monte Carlo analizlerinde Klasik EKF'nin bazı tohumlarda diverje olması, otonom otobüslerde harici outlier rejection ve gating katmanlarının gerekliliğini göstermektedir.
5.  **RL Basitleştirmesi:** Durak yaklaşma kontrolünde kullanılan DQN ajanı basitleştirilmiş 1B kinematik modeli ve bekleme (dwell) ödüllerini temel almaktadır. Sonuçlar gerçek kontrol doğrulaması değil, asenkron lokalizasyon girdisinin kontrole etkisini inceleyen yöntemsel bir ön çalışmadır.
