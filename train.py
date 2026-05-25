from utils.logging_utils import setup_logger
from ultralytics import YOLO
# from pathlib import Path
import argparse
import time
import logging
import yaml
from utils.paths import Paths
from utils.config import TRAIN_CONFIG


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--model", type=str, default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0")

    parser.add_argument("--project", type=str, default="runs")
    parser.add_argument("--name", type=str, default=None)

    return parser.parse_args()


# def setup_logger():
#     log_dir = Paths.LOGS
#     log_dir.mkdir(exist_ok=True)

#     timestamp = time.strftime("%Y%m%d_%H%M%S")
#     log_file = log_dir / f"train_{timestamp}.log"

#     logging.basicConfig(
#         level=logging.INFO,
#         format="%(asctime)s - %(message)s",
#         handlers=[
#             logging.FileHandler(log_file, encoding="utf-8"),
#             logging.StreamHandler()
#         ]
#     )

#     return logging.getLogger(), log_file


def save_experiment(args, save_dir):
    exp_dir = Paths.EXPERIMENTS
    exp_dir.mkdir(exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    exp_file = exp_dir / f"exp_{timestamp}.yaml"

    exp_data = {
        "time": timestamp,
        "model": args.model,
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "save_dir": str(save_dir)
    }

    with open(exp_file, "w") as f:
        yaml.dump(exp_data, f, allow_unicode=True)

    return exp_file


def main():
    args = parse_args()

    logger, log_file = setup_logger()

    # 自动命名实验
    if args.name is None:
        args.name = f"exp_{time.strftime('%m%d_%H%M')}"

    logger.info("===== Training Start =====")
    logger.info(f"Model: {args.model}")
    logger.info(f"Data: {args.data}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch: {args.batch}")
    logger.info(f"Image Size: {args.imgsz}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Experiment Name: {args.name}")

    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,

        # # ===== 关键训练参数 =====
        # lr0=0.001,
        # lrf=0.01,
        # warmup_epochs=3,
        # cos_lr=True,

        # # ===== 数据增强 =====
        # mosaic=1.0,
        # mixup=0.1,
        # copy_paste=0.2,

        # close_mosaic=10,
        # patience=30,

        # # ===== 几何增强 =====
        # degrees=5.0,
        # fliplr=0.5,

        # workers=8

        **TRAIN_CONFIG
    )

    save_dir = results.save_dir

    # 保存实验记录
    exp_file = save_experiment(args, save_dir)

    logger.info("===== Training End =====")
    logger.info(f"Results saved in: {save_dir}")
    logger.info(f"Experiment record saved in: {exp_file}")
    logger.info(f"Log file: {log_file}")


if __name__ == "__main__":
    main()