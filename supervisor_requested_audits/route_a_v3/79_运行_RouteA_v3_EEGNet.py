"""编号79：直接运行 Route A v3 EEGNet 基线。

运行本文件即可，不需要填写命令行参数。训练较慢，支持 checkpoint 续跑。
输出目录由公共工具固定为 08_eegnet。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
path = HERE / "78_公共_RouteA_v3深度基线工具.py"
spec = importlib.util.spec_from_file_location("route_a_v3_deep_common_eegnet", path)
if spec is None or spec.loader is None:
    raise ImportError(path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


if __name__ == "__main__":
    module.run_v3_model("eegnet")
