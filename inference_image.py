import argparse
import cv2
import torch
import numpy as np

# — adjust this to your actual model class —
from model import WeaponDetector  

def load_model(weights_path, device):
    model = YOLO(weights)  
    # if you saved a dict with key 'model_state_dict', change accordingly:
    model.to(device).eval()
    return model

def preprocess(img, img_size):
    """Resize & pad to square, normalize, CHW, [0,1]"""
    h0, w0 = img.shape[:2]
    r = img_size / max(h0, w0)
    new_unpad = (int(w0 * r), int(h0 * r))
    resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    canvas = np.full((img_size, img_size, 3), 114, dtype=np.uint8)
    dw = (img_size - new_unpad[0]) // 2
    dh = (img_size - new_unpad[1]) // 2
    canvas[dh:dh+new_unpad[1], dw:dw+new_unpad[0]] = resized
    # BGR→RGB, HWC→CHW, to float
    tensor = canvas[:, :, ::-1].transpose(2, 0, 1)
    tensor = np.ascontiguousarray(tensor, dtype=np.float32) / 255.0
    return tensor, r, (dw, dh)

def postprocess(pred, conf_thres, orig_shape, r, pad):
    """
    pred: numpy array [N,6] → x1,y1,x2,y2,conf,cls
    returns list of (x1,y1,x2,y2,conf,cls)
    """
    boxes = []
    h0, w0 = orig_shape[:2]
    dw, dh = pad
    for *xyxy, conf, cls in pred:
        if conf < conf_thres:
            continue
        x1, y1, x2, y2 = xyxy
        # unpad & unscale
        x1 = int((x1 - dw) / r)
        y1 = int((y1 - dh) / r)
        x2 = int((x2 - dw) / r)
        y2 = int((y2 - dh) / r)
        boxes.append((x1, y1, x2, y2, float(conf), int(cls)))
    return boxes

def draw_boxes(img, boxes, class_names):
    for x1, y1, x2, y2, conf, cls in boxes:
        label = f"{class_names[cls]} {conf:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img, label, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )
    return img

def main():
    p.add_argument(
        "--names", nargs="+",
        default=["weapon"],
        help="List of class names, e.g. ['pistol','rifle']"
    )
    image = "IMG_1.jpg"
    weights = "best.pt"
    output = "out.jpg"
    img_size = 1024
    conf_thres = 0.25
    device = "cpu"
    img0 = cv2.imread(image)
    if img0 is None:
        raise FileNotFoundError(f"Cannot read image {image}")

    tensor, r, pad = preprocess(img0, img_size)
    tensor = torch.from_numpy(tensor).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(
                tensor,
                device="cpu",
                imgsz=1024,
                conf= 0.25,
                iou= 0.45
            )[0]  # first (and only) batch              # your model’s forward → e.g. a tensor
        pred = pred.cpu().numpy()[0]      # shape [N,6]

    boxes = postprocess(pred, args.conf_thres, img0.shape, r, pad)
    out_img = draw_boxes(img0.copy(), boxes, args.names)

    cv2.imwrite(args.output, out_img)
    print(f"Output saved to {args.output}")

if __name__ == "__main__":
    main()
