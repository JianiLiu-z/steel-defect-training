from pathlib import Path


class ValidationResult:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.infos = []

    def add_error(self, msg):
        self.errors.append(msg)

    def add_warning(self, msg):
        self.warnings.append(msg)

    def add_info(self, msg):
        self.infos.append(msg)

    def summary(self):
        print("\n===== Validation Summary =====")

        for msg in self.errors:
            print(f"[ERROR] {msg}")

        for msg in self.warnings:
            print(f"[WARNING] {msg}")

        for msg in self.infos:
            print(f"[INFO] {msg}")

        print(f"\nErrors: {len(self.errors)}")
        print(f"Warnings: {len(self.warnings)}")
        print(f"Infos: {len(self.infos)}")


def validate_dataset_structure(dataset_dir):
    result = ValidationResult()

    dataset_dir = Path(dataset_dir)

    required_dirs = [
        dataset_dir / "images/train",
        dataset_dir / "images/val",
        dataset_dir / "labels/train",
        dataset_dir / "labels/val",
    ]

    for d in required_dirs:
        if not d.exists():
            result.add_error(f"Missing directory: {d}")
        else:
            result.add_info(f"Found directory: {d}")

    return result