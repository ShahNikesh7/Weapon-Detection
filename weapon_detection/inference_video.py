from pathlib import Path
import cv2
import torch
from ultralytics import YOLO

# ────────────────────────── CLI ──────────────────────────
source = "test/VID_1.mp4"
output = "inf_vid"
weights = "runs/finetune_best/weights/best.pt"
_conf = 0.4
_iou = 0.2
_imgsz = 1024
device = "cuda"

def make_writer(path, fps, size):
    """
    Return an opened cv2.VideoWriter and the final Path.
    Raises RuntimeError if no combination works.
    """
    trials = [
        # container, fourcc, preferred APIs to probe (None → default)
        (".mp4", "avc1", [cv2.CAP_FFMPEG, cv2.CAP_GSTREAMER, None]),
        (".mp4", "mp4v", [cv2.CAP_FFMPEG, None]),
        (".avi", "XVID", [None]),
        (".avi", "MJPG", [None]),            # big but universal
    ]

    for ext, four, apis in trials:
        p = Path(path).with_suffix(ext)
        fourcc = cv2.VideoWriter_fourcc(*four)
        for api in apis:
            api = api if api is not None else 0  # 0 = autodetect/default
            vw = cv2.VideoWriter(str(p), api, fourcc, fps, size, True)
            if vw.isOpened():
                print(f"[INFO] Using {p.suffix}  fourcc={four}  api={api}")
                return vw, p
            vw.release()

    raise RuntimeError(
        "No working codec/container. "
    )
# ────────────────────────── MAIN ─────────────────────────
def main():
    src = Path(source)
    if not src.exists():
        raise FileNotFoundError(src)

    # Load model once
    dev = ("cuda:0" if torch.cuda.is_available() else "cpu")
    model = YOLO(weights).to(dev)

    # OpenCV capture & writer
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {src}")

    w, h = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")              # → .mp4 (AVC)
    writer, out_path = make_writer(output, fps, (w, h))
    #writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    if not writer.isOpened(): raise RuntimeError("VideoWriter failed (codec)")
    print(f"[INFO] Processing {src.name} -> {out_path.name}  ({w}x{h} @ {fps:.2f} fps)")
    frame_idx = 0
    # ── overlay state ─────────────────────────────────────
    last_detections = []   # list of (xyxy, cls, conf)
    no_det_count    = 0
    OVERLAY_FRAMES  = 5
    with torch.no_grad():
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            #  Inference
            results = model(frame,
                            imgsz=_imgsz,
                            conf=_conf,
                            iou=_iou,
                            verbose=False)[0]

            # pull out arrays
            boxes = results.boxes.xyxy.cpu().int().numpy()
            clses = results.boxes.cls.cpu().numpy().astype(int)
            confs = results.boxes.conf.cpu().numpy()

            if boxes.shape[0] > 0:
                # new detections → draw & reset buffer
                last_detections = list(zip(boxes, clses, confs))
                no_det_count = 0

                for xyxy, cls, conf in last_detections:
                    x1, y1, x2, y2 = xyxy
                    label = f"Weapon {conf:.2f}"
                    # draw only the corners
                    color = (0, 0, 255)
                    thickness = 2
                    # length of each corner line in pixels (you can tweak this)
                    corner_len = 20  
                    
                    # top-left
                    cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, thickness)
                    cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, thickness)
                    # top-right
                    cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, thickness)
                    cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, thickness)
                    # bottom-left
                    cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, thickness)
                    cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, thickness)
                    # bottom-right
                    cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, thickness)
                    cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, thickness)
                    cv2.putText(
                        frame, label, (x1, max(y1 - 6, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2,
                        cv2.LINE_AA
                    )

            elif last_detections and no_det_count < OVERLAY_FRAMES:
                # no new detections but still within overlay window
                for xyxy, cls, conf in last_detections:
                    x1, y1, x2, y2 = xyxy
                    label = f"Weapon {conf:.2f}"
                    # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    # draw only the corners
                    color = (0, 0, 255)
                    thickness = 2
                    # length of each corner line in pixels (you can tweak this)
                    corner_len = 20  
                    
                    # top-left
                    cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, thickness)
                    cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, thickness)
                    # top-right
                    cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, thickness)
                    cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, thickness)
                    # bottom-left
                    cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, thickness)
                    cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, thickness)
                    # bottom-right
                    cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, thickness)
                    cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, thickness)
                    cv2.putText(
                        frame, label, (x1, max(y1 - 6, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2,
                        cv2.LINE_AA
                    )
                no_det_count += 1

            else:
                # passed overlay window → clear buffer
                no_det_count += 1
                if no_det_count >= OVERLAY_FRAMES:
                    last_detections = []
                    
            writer.write(frame)
            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"  …{frame_idx} frames", end="\r")

    cap.release()
    writer.release()
    print(f"\n[INFO] Done. Saved -> {out_path.resolve()}")

# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
