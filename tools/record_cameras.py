import os
import subprocess
import time
from datetime import datetime

# ===================== AYARLAR =====================

KAYIT_KLASORU = "/home/user/recordings"

RTSP_URLS = [
    "rtsp://admin:Zxcv2Zxcv2@10.223.70.32:554/Streaming/Channels/101",
    "rtsp://admin:Zxcv2Zxcv2@10.223.70.16:554/Streaming/Channels/101",
    "rtsp://admin:Zxcv2Zxcv2@10.223.70.46:554/Streaming/Channels/101",
    "rtsp://admin:Zxcv2Zxcv2@10.223.70.44:554/Streaming/Channels/101",
    "rtsp://admin:Zxcv2Zxcv2@10.223.70.17:554/Streaming/Channels/101",
    "rtsp://admin:Zxcv2Zxcv2@10.223.70.18:554/Streaming/Channels/101",
    "rtsp://admin:Zxcv2Zxcv2@10.223.70.25:554/Streaming/Channels/101",
    "rtsp://admin:Zxcv2Zxcv2@10.223.70.34:554/Streaming/Channels/101",
    "rtsp://admin:Zxcv2Zxcv2@10.223.70.26:554/Streaming/Channels/101",
    "rtsp://admin:Zxcv2Zxcv2@10.223.70.33:554/Streaming/Channels/101",
    "rtsp://admin:Zxcv2Zxcv2@10.223.70.31:554/Streaming/Channels/101",
    "rtsp://admin:Zxcv2Zxcv2@10.223.70.42:554/Streaming/Channels/101",
    "rtsp://admin:Zxcv2Zxcv2@10.223.70.21:554/Streaming/Channels/101",
    "rtsp://admin:Zxcv2Zxcv2@10.223.70.27:554/Streaming/Channels/101",
    "rtsp://admin:Zxcv2Zxcv2@10.223.70.30:554/Streaming/Channels/101",
    "rtsp://admin:Zxcv2Zxcv2@10.223.70.20:554/Streaming/Channels/101",
    "rtsp://admin:Zxcv2Zxcv2@10.223.70.19:554/Streaming/Channels/101",
    "rtsp://admin:Zxcv2Zxcv2@10.223.70.28:554/Streaming/Channels/101",
    "rtsp://admin:Zxcv2Zxcv2@10.223.70.15:554/Streaming/Channels/101",
    "rtsp://admin:Zxcv2Zxcv2@10.223.70.22:554/Streaming/Channels/101",
    "rtsp://admin:Zxcv2Zxcv2@10.223.70.48:554/Streaming/Channels/101",
    "rtsp://admin:Zxcv2Zxcv2@10.223.70.24:554/Streaming/Channels/101",
    "rtsp://admin:Zxcv2Zxcv2@10.223.70.29:554/Streaming/Channels/101",
    "rtsp://admin:Zxcv2Zxcv2@10.223.70.45:554/Streaming/Channels/101",
    "rtsp://admin:Zxcv2Zxcv2@10.223.70.23:554/Streaming/Channels/101",
]

# ===================================================


def start_recording(urls):
    # Her calistirmada yeni klasor: Video-Kabirhan/yaris_20260303_143022/
    session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(KAYIT_KLASORU, f"yaris_{session_ts}")
    os.makedirs(session_dir, exist_ok=True)

    print(f"\nKayit klasoru: {session_dir}")
    print(f"Toplam {len(urls)} kamera bulundu.")
    print("Kayitlar baslatiliyor...\n")

    # processes: {cam_index: {"p": Popen, "file": str, "restarts": int}}
    processes = {}

    def launch(i, url):
        ts = datetime.now().strftime("%H%M%S")
        restarts = processes[i]["restarts"] if i in processes else 0
        suffix = f"_r{restarts}" if restarts > 0 else ""
        out_file = os.path.join(session_dir, f"kamera_{i:02d}{suffix}_{ts}.mp4")

        cmd = [
            "ffmpeg",
            "-rtsp_transport", "tcp",
            "-i", url,
            "-c", "copy",
            "-movflags", "+faststart",
            "-y",
            out_file,
        ]
        p = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return p, out_file

    # Ilk baslatma — her kamera arasinda 0.5s bekleme (ayni anda 25 baglanti problemi)
    for i, url in enumerate(urls, start=1):
        p, out_file = launch(i, url)
        processes[i] = {"p": p, "file": out_file, "url": url, "restarts": 0}
        print(f"  [{i:02d}/{len(urls)}] Basladi -> kamera_{i:02d}_{session_ts}.mp4")
        time.sleep(0.5)

    print("\n" + "-" * 50)
    print("Tum kameralar kaydediliyor.")
    print("Durdurmak icin CTRL+C basin.")
    print("-" * 50 + "\n")

    try:
        while True:
            time.sleep(3)

            for i, info in list(processes.items()):
                if info["p"].poll() is not None:
                    info["restarts"] += 1
                    print(f"[UYARI] Kamera {i:02d} koptu. Yeniden baslatiliyor... (#{info['restarts']})")
                    time.sleep(2)
                    p, out_file = launch(i, info["url"])
                    processes[i]["p"] = p
                    processes[i]["file"] = out_file

    except KeyboardInterrupt:
        print("\nKayitlar guvenli sekilde kapatiliyor...")
        end_ts = datetime.now().strftime("%H%M%S")

        for i, info in processes.items():
            p = info["p"]
            try:
                p.communicate(input=b"q", timeout=10)
            except Exception:
                p.terminate()

            # Dosya adina bitis saatini ekle
            old = info["file"]
            if os.path.exists(old):
                base, ext = os.path.splitext(old)
                new = f"{base}_END{end_ts}{ext}"
                try:
                    os.rename(old, new)
                except Exception:
                    pass

        print(f"\nTum kayitlar tamamlandi.")
        print(f"Klasor: {session_dir}")


if __name__ == "__main__":
    if not RTSP_URLS:
        print("RTSP listesi bos.")
    else:
        start_recording(RTSP_URLS)
