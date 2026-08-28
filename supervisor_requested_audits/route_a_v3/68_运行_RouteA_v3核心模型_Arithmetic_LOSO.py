"""编号68：Route A v3 Arithmetic participant-level nested LOSO。

固定使用 S01--S15、每人四条 recording。输出为 65.../01_core/loso_arithmetic。
除最终测试预测外，还保存每个外层折内的 participant-level OOF meta 预测。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("route_a_v3_core_tools", HERE / "66_公共_RouteA_v3核心模型工具.py")
if spec is None or spec.loader is None:
    raise ImportError("Cannot import numbered core tools")
tools = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = tools
spec.loader.exec_module(tools)


def main() -> None:
    policy, protocol = tools.load_policy_and_protocol()
    device = tools.choose_device()
    core = tools.load_core_module()
    arithmetic, _stroop, metadata, _manifest = core.load_openbci_eeg_only(tools.PROJECT_ROOT, device)
    cohort = [int(value) for value in policy["loso"]["arithmetic"]["participants"]]
    dataset = tools.subset_bundle(arithmetic, subjects=set(cohort), raw_labels={0, 1, 2, 3})
    output = tools.V3_ROOT / "loso_arithmetic"
    tools.ensure_core_metadata(output, policy, protocol, task="arithmetic_loso", cohort=cohort, target_labels=None)
    (output / "input_counts.json").write_text(
        json.dumps({"windows": int(len(dataset.labels)), "recordings": int(len(set(dataset.recordings.tolist()))), "device": str(device), "metadata_from_loader": metadata}, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tools.run_loso_task(core, dataset, task="arithmetic", cohort=cohort, output=output, protocol=protocol, device=device)
    print(f"Completed v3 Arithmetic LOSO core model: {output}")


if __name__ == "__main__":
    main()
