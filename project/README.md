# Otonom Otobüslerde Adaptif EKF Tabanlı Sensör Füzyonu ve Lokalizasyon İyileştirmesi

Bu proje, kentsel sürüş güzergahlarında GNSS kayıpları, tekerlek kayması ve sinyalizasyon gürültüsü altında otonom otobüslerin konum hassasiyetini korumak amacıyla tasarlanmış **Adaptif Genişletilmiş Kalman Filtresi (AEKF)** uygulamasını içermektedir. Proje ayrıca **çift şeritli yol yapısı**, **yolcu biniş/iniş sistemi**, **kapı güvenlik kamerası**, **6-kamera çevre algı sistemi** ve **RL tabanlı durak yaklaşma kontrolcüsü** ile genişletilmiştir.

Fiziksel modellemede **AKIA Ultra LF12 EV** elektrikli otobüsünün teknik özellikleri baz alınmıştır.

---

## Proje Yapısı

```text
project/
│
├── main.py                         # Simülasyonu koşturan, Monte Carlo ve senaryo analizleri
│
├── simulation/                     # Simülasyon ve Fizik Modelleri
│   ├── bus_model.py                # 2B kinematik model (AKIA EV)
│   ├── sensor_model.py             # GNSS, IMU, Odometri, LiDAR sensör modelleri
│   ├── route_generator.py          # Kapalı çevrim rota + durak durum makinesi
│   ├── passenger_model.py          # [YENİ] Yolcu biniş/iniş davranışı
│   ├── door_system.py              # [YENİ] Kapı durum makinesi + güvenlik
│   ├── stop_manager.py             # [YENİ] Kapsamlı durak işletim yöneticisi
│   ├── lane_model.py               # [YENİ] Şerit merkezi metrikleri
│   └── perception_system.py        # [YENİ] Çevre algı sistemi (6 kamera)
│
├── sensors/                        # [YENİ] Modüler Sensör Paketleri
│   ├── lane_camera.py              # Şerit kamera (60° FOV, 30m menzil)
│   ├── proximity_sensor.py         # Çok yönlü ultrasonik yaklaşım sensörü
│   ├── surround_camera.py          # 6 kameralı 360° çevre algı sistemi
│   └── door_camera.py              # Kapı güvenlik kamerası (2m algı yarıçapı)
│
├── filters/                        # Tahmin/Füzyon Filtreleri
│   ├── ekf.py                      # Klasik EKF (6 durumlu)
│   ├── adaptive_ekf.py             # Residual-tabanlı Adaptif EKF
│   └── ukf.py                      # Unscented Kalman Filtresi
│
├── rl_stop/                        # RL Durak Yaklaşma Kontrolcüsü
│   ├── dqn_agent.py                # DQN ajan (6 boyutlu durum uzayı)
│   └── stop_env.py                 # Durak yaklaşma ortamı (durak durumu dahil)
│
├── evaluation/                     # Performans Değerlendirme
│   ├── metrics.py                  # Lokalizasyon + şerit + durak + yolcu metrikleri
│   └── plots.py                    # Karşılaştırmalı akademik analiz grafikleri
│
├── visualization/                  # Animasyonlar
│   ├── animation_utils.py          # Ortak çizim yardımcıları
│   ├── animate_city_bus.py         # [YENİ] Kapsamlı şehir otobüsü animasyonu
│   ├── animate_lane_keeping.py     # [YENİ] Şerit takip analizi animasyonu
│   ├── animate_bus_stop.py         # Durak dwell animasyonu
│   ├── animate_trajectory.py       # Genel yörünge animasyonu
│   └── animate_bus_focus.py        # Otobüs odaklı takip animasyonu
│
├── outputs/                        # Simülasyon çıktıları
│   ├── figures/                    # Analiz grafikleri (.png)
│   ├── animations/                 # GIF animasyonları
│   └── *.csv                       # Sayısal sonuç dosyaları
│
└── report/
    ├── final_report.tex            # IEEE formatında Türkçe makale (LaTeX)
    └── references.bib              # BibTeX kaynakça
```

---

## Sensör Mimarisi

Proje şu sensörleri simüle eder:

| Sensör | Model | Gürültü | Rol |
|--------|-------|---------|-----|
| **GNSS** | RTK-benzeri | 0.4m / 12m (bozulma) | Primer lokalizasyon |
| **IMU** | MEMS | 0.1°/s gyro, 0.05m/s² acc | Yönelim + ivme |
| **Odometri** | Tekerlek enkoder | 0.08m/s / 1.2m/s (kayma) | Hız tümlevleme |
| **LiDAR** | 180° FOV | 0.1m menzil, 1° açı | Landmark randevu |
| **Şerit Kamerası** | Ön kamera 60° | 2cm lateral | EKF yanal güncelleme |
| **Yaklaşım Sensörü** | Ultrasonik | 3cm | Durak yanaşma |
| **Çevre Kameraları** | 6x kamera 360° | — | Engel/yaya farkındalığı |
| **Kapı Kamerası** | 2m yarıçap | 2% yanlış +, 1% yanlış - | Kapı güvenlik izni |

---

## Durak Durum Makinesi

Otobüsün duraklarda gerçekçi davranışı aşağıdaki durumlarla modellenir:

```
CRUISE → APPROACH_STOP (30m kala)
       → BRAKE_TO_STOP (15m kala)
       → DOOR_OPEN (kapılar açılıyor)
       → ALIGHTING + BOARDING (eş zamanlı)
       → DOOR_CLEARANCE_CHECK (kapı güvenlik kontrolü)
       → DOOR_CLOSE (kapılar kapanıyor)
       → DEPART_STOP → CRUISE
```

**Dinamik dwell süresi:** `t_dwell = 5s + n_binen × 1.5s + n_inen × 1.0s`

---

## Yolcu Senaryoları

| Durak (s) | Bekleyen | İnen |
|-----------|---------|------|
| 200 m | 5 | 2 |
| 600 m | 0 | 4 |
| 950 m | 8 | 1 |

---

## Animasyonlar

| Animasyon | Dosya | İçerik |
|-----------|-------|--------|
| Şehir simülasyonu | `bus_city_simulation.gif` | Çift şerit, 6 kamera, yolcu, kapı |
| Şerit takip | `bus_lane_keeping.gif` | Yanal hata analizi |
| Durak yaklaşma | `bus_stop_dwell.gif` | Yolcu biniş/iniş |
| Yörünge genel | `bus_trajectory_with_stops.gif` | Tam rota |

---

## Çalıştırma

```powershell
# Ana simülasyon (lokalizasyon + Monte Carlo + grafikler)
python -m project.main

# Kapsamlı şehir otobüsü animasyonu
python -m project.visualization.animate_city_bus

# Şerit takip analizi animasyonu
python -m project.visualization.animate_lane_keeping
```

---

## Sayısal Sonuçlar (Nominal — Seed: 42)

### Lokalizasyon Performansı

| Metrik | Klasik EKF | **Adaptif EKF** | UKF | İyileşme (AEKF vs EKF) |
|:---|:---:|:---:|:---:|:---:|
| **Konum RMSE** | 0.2221 m | **0.1556 m** | 0.6914 m | **%29.92** |
| **Maks Konum Hatası** | 1.8882 m | **0.6608 m** | 7.0119 m | **%65.00** |
| **Yönelim RMSE** | 1.8038° | **0.3627°** | 1.1465° | **%79.90** |
| **Durak Konum RMSE** | 0.1190 m | **0.1162 m** | 0.1292 m | **%2.35** |
| **Koridor İhlal Oranı** | %2.36 | **%0.21** | %12.36 | **%91.10** |

### Şerit Takip Performansı

| Metrik | Değer |
|:---|:---:|
| Ortalama Şerit Merkezi Hatası | 0.162 m |
| Maks. Şerit Hatası | 1.837 m |
| Şerit İhlali | 2 kez |
| Şeritte Kalma Başarı Oranı | **%99.89** |

### RL Durak Yaklaşma Kontrolcüsü

| Metrik | Kural Tabanlı | **RL (DQN)** |
|:---|:---:|:---:|
| Ortalama Durak Hatası | 0.38 m | **0.12 m** |
| Aşma Oranı | %15 | **%3** |

---

*Not: 30 seed Monte Carlo testinde Klasik EKF bazı seed'lerde ıraksamıştır (ort. RMSE = 15.79 m). AEKF tüm seed'lerde kararlılığını korumuştur (ort. RMSE = 0.1554 m, std = 0.0079 m).*
