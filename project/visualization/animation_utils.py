import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
from matplotlib.animation import PillowWriter, FFMpegWriter

def draw_bus_body(ax, x, y, theta, length=12.27, width=2.54, color='#00e5ff', alpha=0.4, edgecolor='#00e5ff', label=None):
    """
    Otobüs gövdesini fiziksel boyutlarına göre ölçekli ve yönelimli bir dikdörtgen olarak çizer.
    Rear axle (arka dingil) (x, y) noktasında konumlandırılmıştır.
    """
    # Otobüsün arka dingili, arka tampondan yaklaşık 3.0 m öndedir.
    # Dolayısıyla lokal sol-alt köşe koordinatı (-3.0, -width/2.0) olur.
    rect = patches.Rectangle((-3.0, -width / 2.0), length, width, 
                             linewidth=1.8, edgecolor=edgecolor, 
                             facecolor=color, alpha=alpha, label=label)
    
    # Döndürme ve öteleme matrisini uygula
    t = transforms.Affine2D().rotate(theta).translate(x, y) + ax.transData
    rect.set_transform(t)
    ax.add_patch(rect)
    return rect

def get_shifted_line(points, d):
    """
    Yörünge noktalarını d mesafesi kadar normal vektör yönünde öteler.
    """
    shifted = []
    n_pts = len(points)
    for i in range(n_pts):
        if i < n_pts - 1:
            dx = points[i+1, 0] - points[i, 0]
            dy = points[i+1, 1] - points[i, 1]
        else:
            dx = points[i, 0] - points[i-1, 0]
            dy = points[i, 1] - points[i-1, 1]
        length = np.hypot(dx, dy)
        if length < 1e-6:
            dx, dy = 1.0, 0.0
            length = 1.0
        tx = dx / length
        ty = dy / length
        # normal vektör (sol taraf): (-ty, tx)
        nx = -ty
        ny = tx
        shifted.append([points[i, 0] + d * nx, points[i, 1] + d * ny])
    return np.array(shifted)

def draw_route_corridor(ax, route_points, corridor_width=7.2, color='#222222', alpha=0.15):
    """
    Çift şeritli gidiş-dönüş otoyolunu (genişlik = 7.2 m) asfalt, 
    kenar çizgileri ve sarı kesikli şerit çizgisiyle çizer.
    Otobüs sağ şerit merkezinden (route_points) hareket eder.
    """
    # 1. Asfalt kaplaması (koyu gri kalın şerit)
    asphalt = get_shifted_line(route_points, -1.8)
    ax.plot(asphalt[:, 0], asphalt[:, 1], color='#333333', linewidth=28, alpha=0.20, zorder=0)
    
    # 2. Sol karşı şerit dış sınırı (beyaz)
    left_border = get_shifted_line(route_points, -5.4)
    ax.plot(left_border[:, 0], left_border[:, 1], color='#dddddd', linewidth=1.5, alpha=0.6, zorder=1)
    
    # 3. Orta şerit çizgisi (sarı kesikli çizgi)
    center_line = get_shifted_line(route_points, -1.8)
    ax.plot(center_line[:, 0], center_line[:, 1], color='#ffcb2b', linewidth=1.2, linestyle='--', dashes=(8, 8), alpha=0.8, zorder=1)
    
    # 4. Sağ kendi şeridi dış sınırı (beyaz)
    right_border = get_shifted_line(route_points, +1.8)
    ax.plot(right_border[:, 0], right_border[:, 1], color='#dddddd', linewidth=1.5, alpha=0.6, zorder=1)


def draw_stop_zone(ax, stop_position, stop_radius=10.0, color='orange', alpha=0.15):
    """
    Otobüs durak bölgesinin yanaşma sınırlarını çizer.
    """
    circle = patches.Circle(stop_position, stop_radius, color=color, alpha=alpha, fill=True, zorder=1)
    ax.add_patch(circle)
    
    # Merkezine küçük bir durak işareti koyalım
    ax.plot(stop_position[0], stop_position[1], color='orange', marker='s', markersize=6, markeredgecolor='black')
    return circle

def draw_uncertainty_ellipse(ax, mean, covariance, color='red', alpha=0.15):
    """
    2B konum belirsizliğini temsil eden 2-sigma (%95 güven) hata elipsini çizer.
    """
    vals, vecs = np.linalg.eigh(covariance)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    
    # 2-sigma elips eksen genişlikleri (2 * 2 * sqrt(eigenvalue))
    w = 4.0 * np.sqrt(max(1e-9, vals[0]))
    h = 4.0 * np.sqrt(max(1e-9, vals[1]))
    
    ellipse = patches.Ellipse(xy=mean, width=w, height=h, angle=theta, 
                              edgecolor=color, facecolor=color, alpha=alpha, fill=True, zorder=2)
    ax.add_patch(ellipse)
    return ellipse

def save_animation(anim, output_path, fps=15):
    """
    Animasyonu GIF veya MP4 olarak kaydeder.
    Sistemde ffmpeg varsa ve çıktı mp4 ise FFMpegWriter kullanır. Aksi takdirde PillowWriter (GIF) kullanır.
    """
    dir_name = os.path.dirname(output_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
        
    ext = os.path.splitext(output_path)[1].lower()
    
    if ext == '.mp4':
        try:
            writer = FFMpegWriter(fps=fps, metadata=dict(artist='Antigravity'), bitrate=1800)
            anim.save(output_path, writer=writer)
            print(f"Animasyon MP4 olarak kaydedildi: {output_path}")
        except Exception as e:
            print(f"FFMpegWriter hatası: {e}. GIF formatına dönüştürülüyor...")
            gif_path = output_path.replace('.mp4', '.gif')
            writer = PillowWriter(fps=fps)
            anim.save(gif_path, writer=writer)
            print(f"Animasyon GIF olarak kaydedildi: {gif_path}")
    else:
        # Standart GIF kaydı
        writer = PillowWriter(fps=fps)
        anim.save(output_path, writer=writer)
        print(f"Animasyon GIF olarak kaydedildi: {output_path}")

def interpolate_simulation_data(data, frame_skip=1):
    """
    Dosya boyutunu düşürmek için veriyi frame_skip adımla seyreltir.
    """
    return data[::frame_skip]
