# Qwen SFT Tool-Calling 项目说明

这是一个面向 `Qwen3.5-4B` 的工具调用微调项目，基于 `Transformers + PEFT LoRA` 实现，覆盖了数据预处理、smoke 训练、正式训练、基座评测、LoRA 评测以及若干基础单元测试。

项目当前更偏向“可复现的训练脚本仓库”而不是通用框架，默认目录、数据路径和输出路径都已经在代码里约定好，适合直接在本地或训练机上落地使用。

## 1. 项目能力

- 支持对工具调用数据做去重、切分和 smoke 子集抽样
- 支持基于 LoRA 的 Qwen 指令微调
- 支持对 base model / LoRA model 做工具调用准确率评测
- 支持动态 batch 评测，减少不同 prompt 长度带来的 padding 浪费
- 支持 tokenized dataset cache，加快重复训练时的数据准备
- 支持 smoke 数据集上的快速冒烟验证

## 2. 项目结构

```text
qwen-sft/
├── data/                         # 数据集与统计结果
│   ├── xlam_function_calling_60k.json
│   ├── deduplicated_data.json
│   ├── train.json / val.json / test.json
│   ├── smoke/
│   └── stats/
├── outputs/                      # 训练与评测输出
├── scripts/
│   ├── common/                   # 项目级公共路径、配置辅助函数
│   ├── data_preprocess/          # 数据预处理脚本
│   ├── eval/                     # 评测脚本
│   ├── train/                    # 训练脚本
│   ├── utils/                    # 训练/评测工具函数
│   └── test_qwen.py              # 本地模型可用性快速检查
├── tests/                        # 单元测试
├── requirements.txt
└── README.md
```

### 关键目录说明

- `scripts/common/`
  - 放项目级公共逻辑，比如默认路径、配置通用校验、tool-calling 样本拼装
- `scripts/train/common/`
  - 放训练流程与训练配置的共用逻辑
- `scripts/eval/common/`
  - 放评测流程、动态 batching 配置与公共运行器
- `scripts/data_preprocess/common.py`
  - 放去重、切分、抽样、长度统计等公共函数

## 3. 环境要求

### 推荐环境

- Linux
- Python 3.12
- NVIDIA GPU
- CUDA 12.8+ 驱动

`requirements.txt` 当前是面向 RTX 5090 / CUDA 12.8 环境整理的：

```bash
python -m pip install -r requirements.txt
```

### 模型路径

项目默认会从环境变量 `QWEN_MODEL_PATH` 读取模型目录；如果未设置，则默认寻找：

```text
../models/Qwen3.5-4B
```

建议显式设置：

```bash
export QWEN_MODEL_PATH=/path/to/Qwen3.5-4B
```

## 4. 数据流程

项目默认的数据处理链路如下：

1. 原始数据：`data/xlam_function_calling_60k.json`
2. 去重结果：`data/deduplicated_data.json`
3. 切分结果：`data/train.json`、`data/val.json`、`data/test.json`
4. smoke 子集：`data/smoke/train.json`、`data/smoke/val.json`

### 数据预处理命令

#### 4.1 去重

```bash
python scripts/data_preprocess/1_deduplicate_dataset.py
```

#### 4.2 切分训练/验证/测试集

默认比例为 `0.8 / 0.05 / 0.15`：

```bash
python scripts/data_preprocess/2_split_dataset.py
```

#### 4.3 生成 smoke 数据

默认会从正式训练集和验证集里抽样：

```bash
python scripts/data_preprocess/3_create_smoke_dataset.py
```

#### 4.4 统计样本 token 长度

用于观察训练样本整体长度分布：

```bash
python scripts/data_preprocess/sample_token_stats.py
```

#### 4.5 统计 answer token 长度

用于估算评测时更合适的 `max_new_tokens`：

```bash
python scripts/data_preprocess/answer_token_stats.py
```

## 5. 训练说明

### 训练前快速检查模型是否可用

```bash
python scripts/test_qwen.py
```

这个脚本会加载 tokenizer 和模型，并让模型输出一句简短文本，适合用于确认本地模型路径、依赖和推理环境没问题。

### 5.1 Smoke 训练

推荐先跑 smoke，确认数据、模板、LoRA 注入和保存逻辑都没问题。

```bash
python scripts/train/smoke/train_smoke.py
```

如果你本地习惯用 conda 环境，也可以直接使用仓库自带脚本：

```bash
bash data/smoke/run_smoke_train.sh
```

默认输出目录：

```text
outputs/smoke_lora_train_peft/
```

### 5.2 正式训练

```bash
python scripts/train/LoRA/train_lora.py
```

默认输出目录：

```text
outputs/lora_train_peft/
```

### 5.3 常见训练参数

可以通过 CLI 覆盖默认值，例如：

```bash
python scripts/train/LoRA/train_lora.py \
  --model-path /path/to/Qwen3.5-4B \
  --train-file data/train.json \
  --val-file data/val.json \
  --output-dir outputs/lora_train_peft \
  --max-length 1536 \
  --per-device-train-batch-size 4 \
  --gradient-accumulation-steps 3 \
  --num-epochs 3
```

训练过程会输出：

- 训练配置摘要
- 数据集统计信息
- 学习率与 loss 日志
- 验证集指标
- checkpoint
- 最佳模型与最终 summary

### 5.4 训练输出内容

正式训练或 smoke 训练目录下通常会包含：

- `summary.json`
- `best_eval.json`
- `adapter_model.safetensors`
- `adapter_config.json`
- tokenizer 相关文件
- `checkpoint-*` 子目录

## 6. 评测说明

### 6.1 评测 base model

```bash
python scripts/eval/eval_base_tool_calls.py
```

默认输出目录：

```text
outputs/base_eval/
```

### 6.2 评测正式 LoRA 模型

```bash
python scripts/eval/eval_lora_tool_calls.py
```

默认会读取：

- base model：`QWEN_MODEL_PATH`
- adapter：`outputs/lora_train_peft/`

默认输出目录：

```text
outputs/lora_eval/
```

### 6.3 评测 smoke LoRA 模型

```bash
python scripts/eval/eval_smoke_lora_tool_calls.py
```

默认输出目录：

```text
outputs/smoke_lora_eval/
```

### 6.4 小样本快速评测

仓库里保留了一个 10 条样本的便捷脚本：

```bash
python scripts/eval/eval_base_tool_calls_10.py
```

### 6.5 常见评测参数

```bash
python scripts/eval/eval_lora_tool_calls.py \
  --data-path data/test.json \
  --num-samples 100 \
  --sample-mode random \
  --seed 42 \
  --max-new-tokens 512 \
  --batch-size 28 \
  --max-batch-size 28
```

评测结果会生成两类文件：

- `*.summary.json`：汇总指标
- `*.details.json`：逐样本详情

## 7. 测试

当前仓库内置的是轻量单元测试，主要覆盖：

- 动态 batching 逻辑
- 数据预处理公共函数
- 训练配置校验

运行方式：

```bash
python -m unittest discover -s tests
```

## 8. 工程约定

### 路径约定

项目默认路径集中定义在：

```text
scripts/common/project_paths.py
```

如果你要调整默认数据目录、输出目录或默认模型目录，优先从这里改。

### 公共逻辑放置建议

- 新增路径/默认值：放 `scripts/common/`
- 新增训练共用逻辑：放 `scripts/train/common/`
- 新增评测共用逻辑：放 `scripts/eval/common/`
- 新增数据预处理公共逻辑：放 `scripts/data_preprocess/common.py`

这样可以尽量避免脚本之间复制粘贴逻辑。

## 9. 建议使用顺序

如果你是第一次接手这个仓库，建议按下面顺序走：

1. 配好 `QWEN_MODEL_PATH`
2. 运行 `python scripts/test_qwen.py`
3. 执行数据预处理脚本
4. 跑 `python scripts/train/smoke/train_smoke.py`
5. 跑 `python scripts/eval/eval_smoke_lora_tool_calls.py`
6. 确认 smoke 全链路正常后，再跑正式训练和正式评测

## 10. 备注

- 训练脚本默认要求 GPU，可用性不足时会直接报错
- 项目当前默认围绕 Qwen 工具调用格式构建，数据字段需要至少包含 `query`、`tools`、`answers`
- 部分历史输出文件已保留在仓库中，主要用于参考结果格式，不建议把新的大模型权重长期直接提交到 Git
