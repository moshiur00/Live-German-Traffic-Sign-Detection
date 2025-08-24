

import os
import time
from collections import defaultdict
from typing import Dict, Optional

import cv2
import torch
from ultralytics import YOLO

from .config import AppConfig
from .hud import draw_hud


class DetectorApp:
    """Detector & tracker using Ultralytics YOLO with ByteTrack.

    Modes:
      - live  : webcam/RTSP with tracking
      - video : file with tracking
      - image : image or folder; detection by default, optional tracking across a sequence
    """

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.model = YOLO(cfg.weights)
        self.writer: Optional[cv2.VideoWriter] = None

    @staticmethod
    def _pick_device(arg: str) -> str:
        if arg == "auto":
            return "0" if torch.cuda.is_available() else "cpu"
        return arg

    # ------------------------- STREAM (live/video) -------------------------
    def _open_stream(self):
        return self.model.track(
            source=0 if self.cfg.source == "0" else self.cfg.source,
            stream=True,
            imgsz=self.cfg.imgsz,
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            device=self._pick_device(self.cfg.device),
            tracker=self.cfg.tracker,
            classes=self.cfg.classes,
            max_det=self.cfg.max_det,
            verbose=False,
            persist=True,
            vid_stride=self.cfg.vid_stride,
            show=False,
        )

    def _init_writer(self, frame) -> None:
        if not self.cfg.save or self.writer is not None:
            return
        h, w = frame.shape[:2]
        os.makedirs(os.path.dirname(self.cfg.out), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(self.cfg.out, fourcc, self.cfg.fps, (w, h))

    @staticmethod
    def _active_counts(result) -> Dict[int, int]:
        counts = defaultdict(int)
        if getattr(result, "boxes", None) is not None and getattr(result.boxes, "id", None) is not None:
            cls = result.boxes.cls.cpu().numpy().astype(int)
            for c in cls:
                counts[int(c)] += 1
        return counts

    def _run_stream(self) -> None:
        stream = self._open_stream()
        ema_fps: Optional[float] = None
        t0 = time.time()

        for result in stream:
            frame = result.plot()

            now = time.time()
            inst_fps = 1.0 / max(now - t0, 1e-6)
            ema_fps = inst_fps if ema_fps is None else (0.9 * ema_fps + 0.1 * inst_fps)
            t0 = now

            frame = draw_hud(frame, ema_fps or 0.0, self._active_counts(result), self.model.names)

            if self.cfg.save and self.writer is None:
                self._init_writer(frame)
            if self.writer is not None:
                self.writer.write(frame)

            if self.cfg.show:
                cv2.imshow("German Traffic Sign Detector (YOLO + ByteTrack)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        if self.writer is not None:
            self.writer.release()
        cv2.destroyAllWindows()

    # ----------------------------- IMAGES ---------------------------------
    def _run_images_detect(self) -> None:
        """Detect on an image, glob, or folder and save annotated images."""
        os.makedirs(self.cfg.outdir, exist_ok=True)
        results = self.model.predict(
            source=self.cfg.source,
            imgsz=self.cfg.imgsz,
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            device=self._pick_device(self.cfg.device),
            stream=True,
            verbose=False,
        )
        for i, r in enumerate(results):
            im = r.plot()
            # derive output name from input path when available
            name = os.path.basename(getattr(r, "path", f"frame_{i:05d}.jpg"))
            out_path = os.path.join(self.cfg.outdir, name)
            cv2.imwrite(out_path, im)
            if self.cfg.show:
                cv2.imshow("Detections", im)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        cv2.destroyAllWindows()

    def _run_images_track(self) -> None:
        """Track across a folder (image sequence). Saves annotated images."""
        os.makedirs(self.cfg.outdir, exist_ok=True)
        stream = self.model.track(
            source=self.cfg.source,       # can be folder or pattern e.g. images/*.jpg
            stream=True,
            imgsz=self.cfg.imgsz,
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            device=self._pick_device(self.cfg.device),
            tracker=self.cfg.tracker,
            classes=self.cfg.classes,
            max_det=self.cfg.max_det,
            verbose=False,
            persist=True,
            show=False,
        )
        for i, r in enumerate(stream):
            im = r.plot()
            name = os.path.basename(getattr(r, "path", f"frame_{i:05d}.jpg"))
            out_path = os.path.join(self.cfg.outdir, name)
            cv2.imwrite(out_path, im)
            if self.cfg.show:
                cv2.imshow("Tracked Sequence", im)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        cv2.destroyAllWindows()

    # ----------------------------- DISPATCH -------------------------------
    def run(self) -> None:
        mode = (self.cfg.mode or "live").lower()
        if mode in ("live", "video"):
            self._run_stream()
        elif mode == "image":
            if self.cfg.image_seq_track:
                self._run_images_track()
            else:
                self._run_images_detect()
        else:
            raise ValueError(f"Unknown mode: {self.cfg.mode}")
