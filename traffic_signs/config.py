
from typing import List, Optional


class AppConfig:
    """Runtime configuration for the detector app."""

    def __init__(
        self,
        weights: str = "best.pt",
        source: str = "0",
        imgsz: int = 640,
        conf: float = 0.35,
        iou: float = 0.50,
        tracker: str = "bytetrack.yaml",
        device: str = "auto",
        show: bool = False,
        save: bool = False,
        out: str = "outputs/traffic_sign_tracking.mp4",
        fps: float = 30.0,
        vid_stride: int = 1,
        max_det: int = 300,
        classes: Optional[List[int]] = None,
        # New
        mode: str = "live",                # 'live' | 'video' | 'image'
        outdir: str = "outputs/images",    # used when mode='image'
        image_seq_track: bool = False,      # track across a folder sequence
    ):
        self.weights = weights
        self.source = source
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.tracker = tracker
        self.device = device
        self.show = show
        self.save = save
        self.out = out
        self.fps = fps
        self.vid_stride = vid_stride
        self.max_det = max_det
        self.classes = classes
        self.mode = mode
        self.outdir = outdir
        self.image_seq_track = image_seq_track


def parse_classes(arg: Optional[str]) -> Optional[List[int]]:
    if not arg:
        return None
    out: List[int] = []
    for p in arg.split(","):
        p = p.strip()
        if p.isdigit():
            out.append(int(p))
    return out or None
