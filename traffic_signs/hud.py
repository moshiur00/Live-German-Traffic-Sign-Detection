
import cv2
from typing import Dict, Mapping


def draw_hud(img, fps: float, id_counts: Dict[int, int], names: Mapping[int, str]):
    """Minimal on-frame HUD with FPS and active track counts per class."""
    pad = 8
    lines = [f"FPS: {fps:.1f}"]
    if id_counts:
        lines.append("Active tracks:")
        for cid in sorted(id_counts):
            label = names.get(int(cid), str(cid)) if hasattr(names, "get") else str(cid)
            lines.append(f"  {label}: {id_counts[cid]}")

    (tw, th), _ = cv2.getTextSize(" ".join(lines), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    box_w = min(360, tw + pad * 2)
    box_h = (len(lines) * (th + 6)) + pad
    cv2.rectangle(img, (pad, pad), (pad + box_w, pad + box_h), (0, 0, 0), -1)

    y = pad + th + 4
    for line in lines:
        cv2.putText(img, line, (pad + 6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, lineType=cv2.LINE_AA)
        y += th + 6
    return img
