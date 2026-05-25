from utils.validation import validate_dataset_structure
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="YOLO dataset root")
    return parser.parse_args()


def count_files(path, exts):
    total = 0
    for ext in exts:
        total += len(list(path.glob(f"*{ext}")))
    return total


def main():
    args = parse_args()

    root = Path(args.dataset)

    result = validate_dataset_structure(root)

    result.summary()

    if result.errors:
        raise SystemExit(1)

    image_exts = [".jpg", ".jpeg", ".png", ".bmp"]

    train_img = count_files(root / "images" / "train", image_exts)
    val_img = count_files(root / "images" / "val", image_exts)

    train_label = len(list((root / "labels" / "train").glob("*.txt")))
    val_label = len(list((root / "labels" / "val").glob("*.txt")))

    print("Dataset check result:")
    print("train images:", train_img)
    print("val images:", val_img)
    print("train labels:", train_label)
    print("val labels:", val_label)

    if train_img == 0 or val_img == 0:
        print("[ERROR] Empty train or val images.")
        raise SystemExit(1)

    if train_img != train_label:
        print("[WARNING] train images and labels count mismatch.")

    if val_img != val_label:
        print("[WARNING] val images and labels count mismatch.")

    print("Check finished.")

if __name__ == "__main__":
    main()
    # dataset_dir = "datasets/GC10-DET"

    # result = validate_dataset_structure(dataset_dir)

    # result.summary()