"""编号69：Route A v3 Stroop participant-level nested LOSO。

固定排除 S14、S15，使用 S01--S13 的四条 recording。输出为
65.../01_core/loso_stroop，支持已有 checkpoint 续跑。
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
    _arithmetic, stroop, metadata, _manifest = core.load_openbci_eeg_only(tools.PROJECT_ROOT, device)
    cohort = [int(value) for value in policy["loso"]["stroop"]["participants"]]
    dataset = tools.subset_bundle(stroop, subjects=set(cohort), raw_labels={0, 1, 2, 3})
    output = tools.V3_ROOT / "loso_stroop"
    tools.ensure_core_metadata(output, policy, protocol, task="stroop_loso", cohort=cohort, target_labels=None)
    (output / "input_counts.json").write_text(
        json.dumps({"windows": int(len(dataset.labels)), "recordings": int(len(set(dataset.recordings.tolist()))), "device": str(device), "metadata_from_loader": metadata}, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tools.run_loso_task(core, dataset, task="stroop", cohort=cohort, output=output, protocol=protocol, device=device)
    print(f"Completed v3 Stroop LOSO core model: {output}")


if __name__ == "__main__":
    main()
