import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--images", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--classes", required=True)

    parser.add_argument("--split", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def voc_to_yolo(size, box):
    w, h = size

    xmin, ymin, xmax, ymax = box

    x = ((xmin + xmax) / 2) / w
    y = ((ymin + ymax) / 2) / h

    bw = (xmax - xmin) / w
    bh = (ymax - ymin) / h

    return x, y, bw, bh


def parse_class(cls_name):
    """
    适配GC10:
    3_yueyawan -> 2
    """

    cls_name = cls_name.strip()

    # 处理：
    # 3_yueyawan
    if "_" in cls_name:
        try:
            return int(cls_name.split("_")[0]) - 1
        except:
            return None

    # 处理：
    # 3
    try:
        return int(cls_name) - 1
    except:
        return None


def convert_xml(xml_path, label_path):

    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")

    w = int(size.find("width").text)
    h = int(size.find("height").text)

    lines = []

    objects = root.findall("object")
    

    # 没有目标
    if len(objects) == 0:
        return False

    for obj in objects:

        cls_name = obj.find("name").text
       

        cls_id = parse_class(cls_name)

        # 类别解析失败
        if cls_id is None:
            print(f"[WARNING] invalid class {cls_name} in {xml_path.name}")
            continue

        box = obj.find("bndbox")

        xmin = float(box.find("xmin").text)
        ymin = float(box.find("ymin").text)
        xmax = float(box.find("xmax").text)
        ymax = float(box.find("ymax").text)

        x, y, bw, bh = voc_to_yolo(
            (w, h),
            (xmin, ymin, xmax, ymax)
        )

        lines.append(
            f"{cls_id} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}"
        )

    # 没有有效框
    if len(lines) == 0:
        return False

    with open(label_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return True


def find_image(images_dir, stem):

    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:

        path = images_dir / f"{stem}{ext}"

        if path.exists():
            return path

    return None


def main():

    args = parse_args()

    images_dir = Path(args.images)
    annotations_dir = Path(args.annotations)
    output_dir = Path(args.output)

    xml_files = list(annotations_dir.glob("*.xml"))

    random.seed(args.seed)
    random.shuffle(xml_files)

    split_idx = int(len(xml_files) * args.split)

    train_files = xml_files[:split_idx]
    val_files = xml_files[split_idx:]

    total_images = 0

    for split_name, files in [
        ("train", train_files),
        ("val", val_files)
    ]:

        image_out = output_dir / "images" / split_name
        label_out = output_dir / "labels" / split_name

        image_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)

        for xml_path in files:

            image_path = find_image(
                images_dir,
                xml_path.stem
            )

            if image_path is None:
                print(f"[WARNING] image not found for {xml_path.name}")
                continue

            success = convert_xml(
                xml_path,
                label_out / f"{xml_path.stem}.txt"
            )

            if not success:
                continue

            shutil.copy2(
                image_path,
                image_out / image_path.name
            )

            total_images += 1

    print("Convert finished.")
    print(f"Total valid images: {total_images}")
    print("Output:", output_dir)


if __name__ == "__main__":
    main()