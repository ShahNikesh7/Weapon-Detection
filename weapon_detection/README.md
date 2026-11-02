# Weapon Detection Video

A simple end-to-end demo for running weapon detection on a video file using a finetuned Ultralytics YOLO and OpenCV.  
Given a video as input, the script runs inference frame-by-frame, draws “corner” boxes around detected weapons, and writes out a new annotated video.

---

## Quickstart

1. **Clone the repo**  
   ```bash
   git clone https://github.com/SoundSafe-ai/Weapon_detection_video.git
   cd Weapon_detection_video

2. **Install dependencies** 
    ```bash
    pip install -r requirements.txt

3. **Configure inference parameters**
    ```bash
    source = "test/VID_1.mp4"                       # ← path to your input video
    output = "inf_vid"                              # ← output file base name (no extension)
    weights = "runs/finetune_best/weights/best.pt"  # ← your .pt model weights
    _conf = 0.4                                     # ← confidence threshold (0.0–1.0)
    _iou  = 0.2                                     # ← NMS IoU threshold
    _imgsz = 1024                                   # ← inference image size (px)
    device = "cuda"                                 # ← "cuda" or "cpu"

P.S Keep the image size at 1024 for best results
4. **Run inference**
    ```bash
    python inference_video.py


## How It Works
1. **Model loading**

 - model = YOLO(weights).to(device)

2. **Video I/O**

 - Captures frames via cv2.VideoCapture.

 - Chooses a working codec/container using make_writer().

 - Writes annotated frames with VideoWriter.

3. **Inference loop**

 - Read each frame

 - model(frame, imgsz=_imgsz, conf=_conf, iou=_iou)

 - Extract boxes, confidences

 - Draw “corner” style bounding boxes + confidence label

 - Write frame to output

4. **Overlay smoothing**

If a detection briefly disappears, the last box stays on screen for up to 5 frames (OVERLAY_FRAMES = 5).

## Customization

#### Adjust thresholds

 - _conf (confidence)

 - _iou (IoU for NMS)

 - _imgsz (inference resolution)
