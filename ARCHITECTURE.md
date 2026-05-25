# 钢材表面缺陷检测项目架构说明

## 项目概述

本项目基于 **Ultralytics YOLO** 框架，针对 **GC10-DET** 数据集（10 类钢材表面缺陷检测）进行目标检测模型训练，通过迁移学习微调 YOLO11/YOLO26 预训练模型。

## 目录结构

```
steel-project-no-dataset/
├── configs/                    # 配置文件
│   ├── gc10_classes.txt        # 10 类缺陷名称列表
│   └── gc10_det.yaml           # YOLO 数据集路径与类别定义
├── utils/                      # 工程化工具模块
│   ├── paths.py                #   统一路径管理（Paths 类）
│   ├── config.py               #   训练超参数配置（TRAIN_CONFIG）
│   ├── logging_utils.py        #   模块化日志（setup_logger）
│   └── validation.py           #   数据验证子系统（ValidationResult）
├── scripts/                    # 数据预处理脚本
│   ├── convert_voc_to_yolo.py  #   VOC XML → YOLO 格式转换
│   ├── check_dataset.py        #   数据集完整性校验
│   └── augment_small_classes.py #  小类别离线数据增强
├── auto_train.sh               # 多模型顺序训练脚本
├── experiments/                # 实验元数据（每实验一份 YAML）
├── logs/                       # 训练日志
├── runs_archive/               # 训练参数快照（每实验一份 args.yaml）
├── results/                    # 训练结果可视化（CSV / PR曲线 / 混淆矩阵）
├── weights/                    # 模型权重（按实验隔离，best.pt + last.pt）
├── train.py                    # 训练主入口
├── yolo11s.pt / yolo11m.pt / yolo26n.pt  # 预训练权重
└── requirements.txt            # Python 依赖
```

## 数据流水线

```
原始数据 (VOC XML + 图片)
       │
       ▼
  convert_voc_to_yolo.py     ← VOC 标注 → YOLO 归一化格式，按 80/20 划分训练/验证集
       │
       ▼
  check_dataset.py           ← 校验目录结构、统计图片/标签数量、检查匹配
       │
       ▼
  augment_small_classes.py   ← 对包含类别 6/7/8 的图片做离线增强（翻转、亮度、噪声、旋转）
       │
       ▼
  train.py                   ← 加载预训练权重，在线增强（mosaic/mixup/copy_paste），训练
       │
       ▼
  输出: 模型权重 + metrics.csv + args.yaml + 训练日志
```

## 各模块说明

### train.py — 训练主入口

核心功能：
- 命令行参数解析（模型、轮次、批次、图片尺寸、设备等）
- 自动命名实验 (`exp_MMDD_HHMM`)
- 日志系统：同时输出到文件 (`logs/`) 和控制台
- 实验追踪：自动保存实验元数据到 `experiments/exp_*.yaml`
- 调用 Ultralytics `model.train()` 进行训练

关键超参数：
| 参数 | 值 | 说明 |
|------|-----|------|
| lr0 | 0.001 | 初始学习率 |
| lrf | 0.01 | 最终学习率因子（余弦退火） |
| warmup_epochs | 3 | 预热轮次 |
| mosaic | 1.0 | 马赛克增强 |
| mixup | 0.1 | 混合增强 |
| copy_paste | 0.2 | 复制粘贴增强 |
| close_mosaic | 10 | 最后 10 轮关闭马赛克 |
| patience | 30 | 早停耐心值 |

### scripts/convert_voc_to_yolo.py — VOC 格式转换器

- 解析 Pascal VOC XML 标注文件
- 处理 GC10 特有的类别命名（如 `3_yueyawan`，提取数字前缀并 0 索引化）
- 坐标转换：VOC 绝对坐标 → YOLO 归一化中心坐标
- 按指定比例（默认 8:2）随机划分训练/验证集
- 输出标准 YOLO 目录结构：`images/train`, `images/val`, `labels/train`, `labels/val`

### scripts/check_dataset.py — 数据集校验器

- 检查 `images/train`, `images/val`, `labels/train`, `labels/val` 四个必需目录
- 统计图片与标签数量，标记不匹配或空分割
- 内部调用 `utils/validation.py` 的 `validate_dataset_structure()` 进行结构化校验

### utils/validation.py — 数据验证子系统

- `ValidationResult` 类：分级收集 error / warning / info，所有问题一次性报告
- `validate_dataset_structure()`：检查 YOLO 四目录结构完整性
- 被 `scripts/check_dataset.py` 调用，也可在其他模块中独立使用

### scripts/augment_small_classes.py — 小类别增强

- 目标类别：`[6, 7, 8]`（inclusion 夹杂物、rolled_pit 轧坑、crease 折痕）
- 每张包含目标类别的图片生成 3 个增强副本
- 增强操作：水平翻转、亮度/对比度调整、高斯噪声、小角度旋转（±5°）
- 标签文件直接复制（因增强不改变边界框）

## 模型选择

| 模型 | 大小 | 用途 |
|------|------|------|
| YOLO11s | ~18 MB | 主力模型，7/8 次实验使用 |
| YOLO11m | ~39 MB | 中等模型，1 次实验使用 |
| YOLO26n | ~5 MB | 已下载权重，尚未使用 |

所有训练均使用 COCO 预训练权重进行迁移学习。

## 实验演进路径

1. **基线实验**（20 轮）→ 验证流程可行性
2. **50 轮训练**（YOLO11s / YOLO11m）→ 对比模型规模影响
3. **离线增强实验**（gc10_yolo11s_aug）→ 验证小类别增强效果
4. **优化配置**（gc10_yolo11s_opt1，60 轮）→ 启用 cos_lr、旋转增强、降低 patience
5. **最终训练**（gc10_final，80 轮，800px）→ 提高分辨率 + 更长训练

关键超参数演变：
- 早期：`cos_lr=false`, `degrees=0.0`, `copy_paste=0.1`, `patience=100`
- 后期：`cos_lr=true`, `degrees=5.0`, `copy_paste=0.2`, `patience=30`

## 设计特点

1. **轻量封装**：不定义自定义模型/损失函数，在 Ultralytics 框架上构建薄层编排
2. **两级增强策略**：在线增强（mosaic/mixup/copy_paste，由 `utils/config.py` 集中配置）+ 离线增强（针对少数类），双管齐下应对类别不平衡
3. **工程化重构**：统一路径管理（`utils/paths.py`）、模块化日志（`utils/logging_utils.py`）、配置驱动训练（`utils/config.py`）、可扩展验证框架（`utils/validation.py`）
4. **GC10 数据适配**：转换脚本专门处理 GC10 数据集非标准命名（如 `3_yueyawan`）

## 10 类缺陷名称

| 索引 | 英文名 | 中文含义 |
|------|--------|---------|
| 0 | punching_hole | 冲孔 |
| 1 | welding_line | 焊缝 |
| 2 | crescent_gap | 月牙弯 |
| 3 | water_spot | 水渍 |
| 4 | oil_spot | 油斑 |
| 5 | silk_spot | 丝斑 |
| 6 | inclusion | 夹杂物 |
| 7 | rolled_pit | 轧坑 |
| 8 | crease | 折痕 |
| 9 | waist_folding | 腰折 |

## 依赖项

```
ultralytics          # YOLO 训练框架
opencv-python        # 图像增强
pillow               # 图像处理
matplotlib           # 可视化
pyyaml               # 配置文件解析
tqdm                 # 进度条
pandas               # 数据处理
numpy==1.26.4        # 数值计算（锁定版本）
```
