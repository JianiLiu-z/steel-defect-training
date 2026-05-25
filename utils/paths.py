from pathlib import Path


class Paths:
    # 项目根目录
    ROOT = Path(__file__).resolve().parent.parent

    # 数据目录
    DATASETS = ROOT / "datasets"

    # 配置目录
    CONFIGS = ROOT / "configs"

    # 日志目录
    LOGS = ROOT / "logs"

    # 实验记录
    EXPERIMENTS = ROOT / "experiments"

    # 训练输出
    RUNS = ROOT / "runs"

    # 脚本目录
    SCRIPTS = ROOT / "scripts"

    # 权重目录
    WEIGHTS = ROOT / "weights"

    @staticmethod
    def ensure_dir(path):
        path.mkdir(parents=True, exist_ok=True)
        return path