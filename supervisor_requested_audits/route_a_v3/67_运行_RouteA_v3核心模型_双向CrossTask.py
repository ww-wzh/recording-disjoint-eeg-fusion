"""编号67：Route A v3 核心模型双向 cross-task 重跑。

固定规则：S01--S13；source 使用 natural/low/mid/high 四条 recording，
target 只使用 low/mid/high 三条 recording。该脚本直接运行，不需要命令行参数，
支持已有 checkpoint 续跑。输出位置为 65.../01_core/cross_task。
"""

from __future__ import annotations

import importlib.util
import json
import sys

HERE = __import__("pathlib").Path(__file__).resolve().parent
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
    arithmetic, stroop, metadata, _manifest = core.load_openbci_eeg_only(tools.PROJECT_ROOT, device)
    cohort = set(int(value) for value in policy["cross_task"]["participants"])
    source_labels = set(int(value) for value in policy["cross_task"]["source_recordings"])
    target_labels = set(int(value) for value in policy["cross_task"]["target_recordings"])
    arithmetic_source = tools.subset_bundle(arithmetic, subjects=cohort, raw_labels=source_labels)
    stroop_source = tools.subset_bundle(stroop, subjects=cohort, raw_labels=source_labels)
    arithmetic_target = tools.subset_bundle(arithmetic, subjects=cohort, raw_labels=target_labels)
    stroop_target = tools.subset_bundle(stroop, subjects=cohort, raw_labels=target_labels)
    output = tools.V3_ROOT / "cross_task"
    tools.ensure_core_metadata(output, policy, protocol, task="cross_task", cohort=sorted(cohort), target_labels=sorted(target_labels))
    (output / "input_counts.json").write_text(
        json.dumps(
            {
                "arithmetic_source_windows": int(len(arithmetic_source.labels)),
                "stroop_source_windows": int(len(stroop_source.labels)),
                "arithmetic_target_windows": int(len(arithmetic_target.labels)),
                "stroop_target_windows": int(len(stroop_target.labels)),
                "source_recordings_per_direction": 13 * 4,
                "target_recordings_per_direction": 13 * 3,
                "device": str(device),
                "metadata_from_loader": metadata,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    tools.run_cross_task(
        core,
        arithmetic_source,
        stroop_source,
        arithmetic_target,
        stroop_target,
        output=output,
        protocol=protocol,
        device=device,
    )
    print(f"Completed v3 cross-task core model: {output}")


if __name__ == "__main__":
    main()
