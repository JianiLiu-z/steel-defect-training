# Training Module README

## 项目职责

本模块负责钢材表面缺陷检测（GC10-DET）的完整训练流程，涵盖数据准备到模型产出的全链路：

| 职责 | 说明 |
|------|------|
| **数据转换** | VOC XML 标注 → YOLO 归一化格式，自动划分训练/验证集 |
| **数据增强** | 在线增强（mosaic/mixup/copy_paste）+ 离线增强（针对小类别的翻转/亮度/噪声/旋转） |
| **数据校验** | 训练前自动检查目录结构、图片-标签匹配、类别有效性 |
| **YOLO 训练** | 基于 Ultralytics 框架，支持 YOLO11s/m、YOLO26n 多模型迁移学习 |
| **实验管理** | 训练参数集中配置、超参快照自动存档、实验元数据持久化 |
| **多模型管理** | 权重按实验名隔离存储，结果（CSV/图表/混淆矩阵）分目录归档 |

---

## 工程化优化

本模块针对原始脚本的典型工程问题进行了系统性重构，核心成果如下：

### 1. Unified Path Management（统一路径管理）

**问题**：原始脚本中路径散落各处，硬编码绝对路径（如 `/root/autodl-tmp/`），换环境即失效。

**方案**：通过 `utils/paths.py` 中的 `Paths` 类集中定义所有路径，以 `__file__` 为锚点自动推导项目根目录：

```
Paths.ROOT          → 项目根目录（自动推导）
Paths.DATASETS      → 数据集目录
Paths.CONFIGS       → 配置文件目录
Paths.LOGS          → 日志目录
Paths.WEIGHTS       → 模型权重目录
Paths.RUNS          → 训练输出目录
Paths.SCRIPTS       → 工具脚本目录
Paths.EXPERIMENTS   → 实验记录目录
```

**效果**：项目可部署在任意路径，无需修改任何硬编码。

### 2. Validation Subsystem（数据验证子系统）

**问题**：数据校验逻辑分散在转换和训练脚本中，一个错误就中断，无法获得完整问题报告。

**方案**：通过 `utils/validation.py` 构建可扩展的验证框架：

- `ValidationResult` — 结果分级收集（error / warning / info），收集所有问题后一次性报告
- `validate_dataset_structure()` — 检查 YOLO 四目录结构完整性

**效果**：训练前一次性获得所有数据问题，避免训练到一半才发现数据缺失。

### 3. Modular Logging（模块化日志）

**问题**：每个脚本重复配置 `logging.basicConfig()`，格式不统一，日志无法持久化。

**方案**：通过 `utils/logging_utils.py` 中的 `setup_logger()` 提供一行式日志配置：

```python
logger, log_file = setup_logger()
# → 自动创建 logs/train_YYYYMMDD_HHMMSS.log
# → 同时输出到文件和控制台
```

**效果**：所有模块共享统一的日志格式和输出策略，零配置即可持久化。

### 4. Config-based Training（配置驱动训练）

**问题**：训练超参数硬编码在 `train.py` 中，调参需改核心代码，多实验之间参数混叠难以追踪。

**方案**：通过 `utils/config.py` 中的 `TRAIN_CONFIG` 字典集中管理所有超参数：

```python
TRAIN_CONFIG = {
    "lr0": 0.001,
    "lrf": 0.01,
    "warmup_epochs": 3,
    "cos_lr": True,
    "mosaic": 1.0,
    "mixup": 0.1,
    "copy_paste": 0.2,
    "close_mosaic": 10,
    "patience": 30,
    "degrees": 5.0,
    "fliplr": 0.5,
    "workers": 8,
}
```

训练时通过 `**TRAIN_CONFIG` 解包注入，调参只需改一个文件。

### 5. Experiment Tracking（实验追踪）

**方案**：每次训练自动完成三件事：
1. **运行参数快照**：Ultralytics 自动保存完整 `args.yaml` 到 `runs_archive/<exp_name>/`
2. **实验元数据**：`train.py` 记录时间戳、模型、数据配置等关键字段到 `experiments/exp_*.yaml`
3. **训练日志**：完整日志持久化到 `logs/train_*.log`

**效果**：可追溯每次实验的完整上下文，复现时只需查看对应的 `args.yaml`。

---

## 训练命令

### 环境准备

```bash
pip install -r requirements.txt
```

### 单次训练

```bash
# 基础训练（YOLO11s, 20 epochs）
python train.py --data configs/gc10_det.yaml --model yolo11s.pt --epochs 20 --batch 8

# 中等模型（YOLO11m, batch 减半以适配显存）
python train.py --data configs/gc10_det.yaml --model yolo11m.pt --epochs 50 --batch 4

# 最终训练（800px 分辨率, 80 epochs）
python train.py --data configs/gc10_det.yaml --model yolo11s.pt --epochs 80 --imgsz 800 --batch 4

# 指定实验名称
python train.py --data configs/gc10_det.yaml --model yolo11s.pt --epochs 50 --name my_experiment
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--data` | str | **必填** | 数据集配置文件路径 |
| `--model` | str | `yolo11n.pt` | 预训练权重路径 |
| `--epochs` | int | `100` | 训练轮次 |
| `--imgsz` | int | `640` | 输入图片尺寸 |
| `--batch` | int | `16` | 批次大小 |
| `--device` | str | `"0"` | GPU 设备编号 |
| `--project` | str | `"runs"` | 输出根目录 |
| `--name` | str | 自动生成 | 实验名称（默认 `exp_MMDD_HHMM`） |

### 超参数修改

训练超参数不通过命令行传入，统一在 `utils/config.py` 的 `TRAIN_CONFIG` 字典中修改。修改后无需改动 `train.py` 代码。

### 历史实验记录

| 实验名称 | 模型 | Epochs | Batch | ImgSz | 关键配置 |
|----------|------|--------|-------|-------|----------|
| `gc10_baseline` | yolo11s | 20 | 8 | 640 | cos_lr=false, degrees=0, copy_paste=0.1 |
| `gc10_yolo11s_50e` | yolo11s | 50 | 8 | 640 | cos_lr=false, degrees=0, copy_paste=0.1 |
| `gc10_yolo11m_50e` | yolo11m | 50 | 4 | 640 | cos_lr=false, degrees=0, copy_paste=0.1 |
| `gc10_yolo11s_aug` | yolo11s | 50 | 8 | 640 | 离线增强数据, cos_lr=false |
| `gc10_final` | yolo11s | 80 | 4 | 800 | cos_lr=true, degrees=5.0, copy_paste=0.2 |

---

## 数据预处理脚本

| 脚本 | 用途 | 使用方式 |
|------|------|----------|
| `scripts/convert_voc_to_yolo.py` | VOC XML 标注 → YOLO 格式，80/20 划分训练/验证集 | `python scripts/convert_voc_to_yolo.py --images <dir> --annotations <dir> --output <dir> --classes <dir>` |
| `scripts/check_dataset.py` | 校验 YOLO 数据集目录结构、统计图片与标签数量、检查匹配 | `python scripts/check_dataset.py --dataset <path>` |
| `scripts/augment_small_classes.py` | 对包含少数类（6/7/8）的图片做离线增强，每张生成 3 份副本 | 直接修改脚本中的 `IMAGE_DIR` / `LABEL_DIR` 后运行 |

---

## 目录结构

```
steel-project-no-dataset/
│
├── train.py                         # 训练主入口
├── requirements.txt                 # Python 依赖
│
├── yolo11s.pt                       # YOLO11 small 预训练权重 (~18 MB)
├── yolo11m.pt                       # YOLO11 medium 预训练权重 (~39 MB)
├── yolo26n.pt                       # YOLO26 nano 预训练权重 (~5 MB)
│
├── configs/                         # 配置文件
│   ├── gc10_det.yaml                #   数据集路径 + 10 类名称定义
│   └── gc10_classes.txt             #   类别名称参考（纯文本）
│
├── utils/                           # 工程化工具模块
│   ├── __init__.py                  #   包初始化
│   ├── paths.py                     #   统一路径管理（Paths 类）
│   ├── config.py                    #   训练超参数配置（TRAIN_CONFIG）
│   ├── logging_utils.py             #   模块化日志（setup_logger）
│   └── validation.py                #   数据验证子系统（ValidationResult）
│
├── scripts/                         # 数据预处理脚本
│   ├── convert_voc_to_yolo.py       #   VOC XML → YOLO 格式转换器
│   ├── check_dataset.py             #   数据集完整性校验
│   └── augment_small_classes.py     #   小类别离线数据增强
│
├── auto_train.sh                    # 多模型顺序训练脚本
│
├── experiments/                     # 实验元数据（每实验一份 YAML）
│   ├── exp_20260524_124307.yaml     #   gc10_baseline
│   ├── exp_20260524_132244.yaml     #   gc10_yolo11s_50e
│   ├── exp_20260524_141717.yaml     #   gc10_yolo11m_50e
│   ├── exp_20260524_154416.yaml     #   gc10_yolo11s_aug
│   └── exp_20260524_175502.yaml     #   gc10_final
│
├── logs/                            # 训练日志
│   ├── yolo11s_baseline_20e.log
│   ├── yolo11s_50e.log
│   ├── yolo11m_50e.log
│   ├── yolo11s_augmented.log
│   └── yolo11s_final_80e_800img.log
│
├── runs_archive/                    # 训练参数快照（每实验一份 args.yaml）
│   ├── gc10_baseline/args.yaml
│   ├── gc10_yolo11s_50e/args.yaml
│   ├── gc10_yolo11m_50e/args.yaml
│   ├── gc10_yolo11s_aug/args.yaml
│   └── gc10_final/args.yaml
│
├── results/                         # 训练结果可视化（CSV / 曲线 / 混淆矩阵 / 批次预览）
│   ├── gc10_baseline/
│   ├── gc10_yolo11s_50e/
│   ├── gc10_yolo11m_50e/
│   ├── gc10_yolo11s_aug/
│   └── gc10_final/
│
└── weights/                         # 模型权重（按实验隔离）
    ├── gc10_baseline/weights/
    │   ├── best.pt
    │   └── last.pt
    ├── gc10_yolo11s_50e/weights/
    │   ├── best.pt
    │   └── last.pt
    ├── gc10_yolo11m_50e/weights/
    │   ├── best.pt
    │   └── last.pt
    ├── gc10_yolo11s_aug/weights/
    │   ├── best.pt
    │   └── last.pt
    └── gc10_final/weights/
        ├── best.pt
        └── last.pt
```
