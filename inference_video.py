"""
inference_realtime.py

Download a YouTube video and run YOLOv8 inference on it in real time,
displaying the live annotated stream in an OpenCV window.
Press ‘q’ to quit.
"""

import time
from pathlib import Path
import torch
import cv2
from yt_dlp import YoutubeDL
from ultralytics import YOLO

url = "https://www.youtube.com/watch?v=3qjFbTuvKgU"  # Kid 
#url = "https://www.youtube.com/watch?v=UbCVoeTWTiA"

#weights = "checkpoint1_wd\\first_train_runs\\weapons_jupyter3\\weights\\best.pt"
#weights = "runs_wblur\\runs\\finetune_wblur_adamw\\weights\\best.pt"
weights = "Bestsofar\\runs\\finetune_gun_random5\\weights\\best.pt"
skip = 0
_conf = 0.4
_iou = 0.45
_imgsz = 1024

def get_stream_url(youtube_url: str) -> str:
    """Use yt‑dlp to extract the best mp4 stream URL (no download)."""
    ydl_opts = {
        # Prefer 720p or 480p progressive MP4 with H.264 (avc1) video
        "format": (
            "bestvideo[ext=mp4][vcodec^=avc1][height<=720]"
            "+bestaudio[ext=m4a]/best[ext=mp4]/best"
        ),
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
    # Some entries bundle audio+video into `url`; others split into formats list:
    if "url" in info:
        return info["url"]
    # fallback for live streams / dash manifests
    formats = info.get("formats", [])
    for f in reversed(formats):
        if f.get("ext") == "mp4" and f.get("url"):
            return f["url"]
    raise RuntimeError("Could not find an mp4 stream URL")

def main():

    # 1) Download YouTube video
    print("[INFO] Resolving stream URL…")
    stream_url = get_stream_url(url)
    print(f"[INFO] Stream URL obtained.")

    # 2) Load YOLOv8 model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model  = YOLO(weights).to(device)        # bind model once

    # 3) OpenCV capture & window
    cap = cv2.VideoCapture(str(stream_url))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {stream_url}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    delay  = int(1000 / fps)
    frame_idx = 0

    cv2.namedWindow("YOLOv8 Weapons (Press 'q' to quit)", cv2.WINDOW_NORMAL)

    last_detections   = []  # list of (xyxy, cls, conf)
    no_det_count      = 0
    OVERLAY_FRAMES    = 3

    print("[INFO] Starting real‑time inference. Press 'q' to exit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                # quick retry helps with transient HLS stalls
                for _ in range(3):
                    time.sleep(0.03)
                    ok, frame = cap.read()
                    if ok:
                        break
                if not ok:
                    print("[INFO] Stream ended or cannot read frames.")
                    break

            frame_idx += 1
            if skip > 1 and frame_idx % skip:
                if cv2.waitKey(delay) & 0xFF == ord("q"):
                    break
                continue

            with torch.no_grad():
                results = model(
                    frame,
                    imgsz=_imgsz,
                    conf=_conf,
                    iou=_iou,
                    verbose=False
                )[0]

            
            # extract boxes
            boxes_xyxy = results.boxes.xyxy.cpu().int().numpy()
            classes    = results.boxes.cls.cpu().numpy().astype(int)
            confs      = results.boxes.conf.cpu().numpy()

            if boxes_xyxy.shape[0] > 0:
                # we have detections → draw them and reset overlay buffer
                last_detections = list(zip(boxes_xyxy, classes, confs))
                no_det_count = 0

                for xyxy, cls, conf in last_detections:
                    x1,y1,x2,y2 = xyxy
                    label = f"{model.names[cls]} {conf:.2f}"
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
                    cv2.putText(
                        frame, label, (x1, y1-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,255), 2,
                        cv2.LINE_AA
                    )

            elif last_detections and no_det_count < OVERLAY_FRAMES:
                # no current detections, but overlay last ones
                for xyxy, cls, conf in last_detections:
                    x1,y1,x2,y2 = xyxy
                    label = f"{model.names[cls]} {conf:.2f}"
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
                    cv2.putText(
                        frame, label, (x1, y1-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,255), 2,
                        cv2.LINE_AA
                    )
                no_det_count += 1

            else:
                # no overlay if we've gone OVERLAY_FRAMES with no detections
                no_det_count += 1
                if no_det_count >= OVERLAY_FRAMES:
                    last_detections = []

            # show
            cv2.imshow("YOLOv8 Weapons (Press 'q' to quit)", frame)
            if cv2.waitKey(delay) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Exited. Cleaned up temporary files.")

if __name__ == "__main__":
    main()
