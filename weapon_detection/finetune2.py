from ultralytics import YOLO

model = YOLO("runs/finetune/weights/best.pt")

metrics2 = model.train(
    data="data/data.yaml",
    epochs=70,               # more training
    imgsz=1024,               # higher-res
    batch=16,                 # smaller batch
    optimizer="SGD",        # switch optimizer
    lr0=0.0003,                # lower initial LR
    lrf=0.0003,                # smoother decay
    weight_decay=0.0009,      # light regularization
    warmup_epochs=5,
    warmup_momentum=0.85,
    warmup_bias_lr=0.05,
    dropout=0.15,
    project="runs",
    name="finetune_new"
)

print(metrics2)
