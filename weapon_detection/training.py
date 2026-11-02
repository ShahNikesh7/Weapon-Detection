from ultralytics import YOLO
model = YOLO("yolov8m.pt")                    
model.train(
    data="data/data.yaml",                    # your val/train split
    epochs=100,
    imgsz=1024,
    batch = 32,
    optimizer = "SGD",
    lr0 = 0.01,
    lrf=0.0001,
    momentum=0.937,
    weight_decay = 0.0005,
    warmup_epochs=5,
    warmup_momentum=0.8,
    warmup_bias_lr = 0.1,
    project="runs",
    name="weapons_jupyter"
)