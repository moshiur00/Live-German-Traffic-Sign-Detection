
# 🚦 Live German Traffic Sign Detector (YOLO11 + ByteTrack)

Computer Vision project that detects **German traffic signs** in real time and tracks them across frames using **YOLO11** for object detection and **ByteTrack** for multi-object tracking.

Supports:
- ✅ **Live webcam / RTSP feeds**  
- ✅ **Video files**  
- ✅ **Images or entire folders** (with optional sequence tracking)  

---

## ✨ Features
- **Object Detection** on images, videos, and live streams.  
- **Multi-object Tracking** using ByteTrack (stable IDs across frames).  
- **HUD Overlay** showing FPS and active tracked objects.  
- **CLI Options** for switching between modes and outputs.  
- **Configurable** thresholds (`conf`, `iou`, tracker params).  

---

## 📂 Project Structure
```
Live German Traffic Sign Detection/
├─ run.py                 # Entry point (CLI)
├─ bytetrack.yaml         # Tracker config (tunable)
├─ traffic_signs/
│  ├─ __init__.py
│  ├─ app.py              # Core detection/tracking logic
│  ├─ config.py           # Config class + CLI mapping
│  ├─ hud.py              # On-screen overlay
└─ outputs/               # Saved videos / images
```

---

## ⚙️ Installation

### 1. Clone repo
```bash
https://github.com/moshiur00/Live-German-Traffic-Sign-Detection.git
cd Live-German-Traffic-Sign-Detection
```

### 2. Setup Python environment
Python ≥ 3.9 recommended.  
Install [PyTorch](https://pytorch.org/get-started/locally/) (GPU or CPU build depending on your system). Then:

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### Live camera
```bash
python run.py --mode live --source 0 --show
```
- `--source 0` = default webcam  
- RTSP/HTTP stream example:  
  ```bash
  python run.py --mode live --source "rtsp://user:pass@ip:554/stream" --show
  ```

### Video file
```bash
python run.py --mode video --source test_video/test_video.mp4 --show --save --out outputs/run.mp4
```

### Images (detection)
```bash
python run.py --mode image --source path/to/img_or_folder --outdir outputs/images --show
```

### Image sequence tracking
```bash
python run.py --mode image --source frames/*.jpg --image-seq-track --outdir outputs/seq_tracked --show
```

---

## 🎯 Options

| Flag               | Description |
|--------------------|-------------|
| `--mode`           | `live` (webcam/RTSP), `video` (video file), `image` (images/folder) |
| `--weights`        | Path to trained YOLO weights (e.g. `runs/detect/gts_yolo11m/weights/best.pt`) |
| `--source`         | Camera index, file path, folder, or URL |
| `--conf`           | Detection confidence threshold (default `0.35`) |
| `--iou`            | IoU threshold for NMS (default `0.50`) |
| `--tracker`        | Path to tracker config (`bytetrack.yaml`) |
| `--device`         | `auto`, `cpu`, or CUDA index (e.g. `0`) |
| `--show`           | Show annotated window |
| `--save`           | Save annotated output video |
| `--out`            | Output video path (for live/video) |
| `--outdir`         | Output folder (for images) |
| `--fps`            | Output FPS for saved video |
| `--image-seq-track`| Track across image sequence instead of per-image detection |

---

## 🛠️ Training Weights
This project was fine-tuned using the [GTSDB - German Traffic Sign Detection Benchmark Computer Vision Model](https://universe.roboflow.com/mohamed-traore-2ekkp/gtsdb---german-traffic-sign-detection-benchmark).  

If you retrain in Google Colab:
1. Export your `best.pt` after training.  
2. Place it into `runs/detect/.../weights/best.pt`.  
3. Use it with `--weights` flag.  

---

## ⚡ Tracker Config (bytetrack.yaml)
The tracker can be tuned for small traffic signs. Example:

```yaml
tracker_type: bytetrack

# Core thresholds
track_high_thresh: 0.45
track_low_thresh: 0.10
new_track_thresh: 0.55

# Association & lifecycle
match_thresh: 0.8
track_buffer: 40
min_box_area: 20
fuse_score: true

# Detector guardrails
conf_thres: 0.35
iou_thres: 0.5
frame_rate: 30
```

---

## 📊 Example Results

### Live Detection & Tracking
*(insert GIF or screenshot here)*  

### Video Processing
*(insert before/after screenshot here)*  

---

## 📜 License
MIT License – free to use, share, and modify.


