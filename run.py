

import argparse
from traffic_signs.config import AppConfig, parse_classes
from traffic_signs import DetectorApp


def main():
    p = argparse.ArgumentParser("German Traffic Sign Detector (YOLO + ByteTrack)")
    p.add_argument("--mode", choices=["live", "video", "image"], default="live",
                   help="What to run: live webcam/rtsp, video file, or image(s)")

    p.add_argument("--weights", default="runs/detect/gts_yolo11m/weights/best.pt")
    p.add_argument("--source", default="0",
                   help="0 (webcam), path/URL to video, or image/dir/glob when mode=image")

    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--iou", type=float, default=0.50)
    p.add_argument("--tracker", default="bytetrack.yaml")
    p.add_argument("--device", default="auto")

    p.add_argument("--show", action="store_true", default=True)
    p.add_argument("--save", action="store_true", default=True)
    p.add_argument("--out", default="outputs/traffic_sign_tracking.mp4",
                   help="Output video path for live/video modes")

    p.add_argument("--outdir", default="outputs/images",
                   help="Output folder for mode=image")

    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--vid-stride", type=int, default=1)
    p.add_argument("--max-det", type=int, default=300)
    p.add_argument("--classes", default=None, help="Comma-separated class indices to keep")

    p.add_argument("--image-seq-track", action="store_true",
                   help="When mode=image, track across a folder/sequence instead of pure detection")

    a = p.parse_args()

    cfg = AppConfig(
        weights=a.weights,
        source=a.source,
        imgsz=a.imgsz,
        conf=a.conf,
        iou=a.iou,
        tracker=a.tracker,
        device=a.device,
        show=a.show,
        save=a.save,
        out=a.out,
        fps=a.fps,
        vid_stride=a.vid_stride,
        max_det=a.max_det,
        classes=parse_classes(a.classes),
        mode=a.mode,
        outdir=a.outdir,
        image_seq_track=a.image_seq_track,
    )

    # Friendly log
    print(f"[INFO] mode={cfg.mode} source={cfg.source} show={cfg.show} save={cfg.save}")

    DetectorApp(cfg).run()


if __name__ == "__main__":
    main()

