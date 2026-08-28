# Route A v3 当前进度

## 已完成

1. `64_冻结_RouteA_v3协议.json`
   - 冻结纯 EEG、重复文件排除、S01--S13 cross-task、Arithmetic LOSO 和 Stroop LOSO 规则。
2. `65_运行_RouteA_v3数据策略与哈希预检.py`
   - 已实际运行并通过。
   - 120/120 条原始 recording 均成功移除 packet counter，保留 8 个 EEG 信号列。
   - 发现并记录 16 组逐字节重复文件。
   - 预计 cross-task 78 条、Arithmetic LOSO 60 条、Stroop LOSO 52 条目标 recording。
3. `66_公共_RouteA_v3核心模型工具.py`
   - 提供统一的数据过滤、nested 训练、LOSO OOF 保存和断点续跑功能。

## 现在运行

按下面顺序在 PyCharm 中直接运行。每个文件都不需要填写命令行参数：

1. `67_运行_RouteA_v3核心模型_双向CrossTask.py`
2. `68_运行_RouteA_v3核心模型_Arithmetic_LOSO.py`
3. `69_运行_RouteA_v3核心模型_Stroop_LOSO.py`
4. `71_汇总并校验_RouteA_v3核心模型输出.py`
5. `72_冻结_RouteA_v3核心预测.py`
6. `73_运行_RouteA_v3_DASF硬门控.py`

三个脚本都使用 GPU（如果当前 PyTorch 能看到 CUDA），中途停止后重新运行会跳过已经完整写入的 checkpoint。
首次运行会建立新的纯8通道特征缓存，不能使用旧 `route_a` 缓存。

## 输出位置

所有新结果只写入本目录下的 `01_core`：

- `01_core/cross_task`
- `01_core/loso_arithmetic`
- `01_core/loso_stroop`

旧的 `route_a/results`、`route_a/frozen` 和旧的 53--57 号结果均已因通道错误和重复文件问题作废，不能混入新统计或论文。

## 运行完成判据

三个脚本完成后，应分别存在：

- cross-task：26 个 participant-direction cells × 5 seeds；每个目标 cell 只有 3 条 recording。
- Arithmetic LOSO：15 个 outer participants × 5 seeds；每折 4 条 recording。
- Stroop LOSO：13 个 outer participants × 5 seeds；每折 4 条 recording。

完成后不要先运行旧的 17--32 号脚本。把三个输出目录的最后一屏或 `raw_seed_predictions.csv` 行数发给我，我会继续生成 v3 的 DASF、CB-SF、Riemannian、EEGNet、EEG-Conformer、固定融合/stacking 和最终统计脚本。

目前 71 和 72 已经完成。下一步运行 73；它会重新计算 Cross-task 的 recording-disjoint OOF gate 证据，可能比 72 耗时明显更长。
