TRAIN_CONFIG = {
    # ===== 训练参数 =====
    "lr0": 0.001,
    "lrf": 0.01,
    "warmup_epochs": 3,
    "cos_lr": True,

    # ===== 数据增强 =====
    "mosaic": 1.0,
    "mixup": 0.1,
    "copy_paste": 0.2,

    # ===== Mosaic关闭 =====
    "close_mosaic": 10,

    # ===== Early Stop =====
    "patience": 30,

    # ===== 几何增强 =====
    "degrees": 5.0,
    "fliplr": 0.5,

    # ===== DataLoader =====
    "workers": 8,
}