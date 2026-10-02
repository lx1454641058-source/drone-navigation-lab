"""来源：本项目原创。运行离散决策场景，生成独立的本地回放页面。"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from .mission import run_mission
from .scenarios import scenarios


def main() -> None:
    parser = argparse.ArgumentParser(description="无人机视觉与导航开发实验")
    parser.add_argument("--output", type=Path, help="新建输出目录；已存在时拒绝覆盖")
    parser.add_argument("--mode", choices=("baseline", "vision", "camera", "explore", "motion", "localization", "planning", "delivery", "physics", "coupled", "descent", "realvision", "scale"), default="baseline",
                        help="baseline=真值；vision=图像；camera=透视；explore=探索；motion=制动；localization=定位误差；planning=可停止路线对照；delivery=视觉降落检查；physics=独立物理台架；coupled=视觉与物理联调；descent=视觉下降；realvision=真实航拍识别；scale=分块尺度对照")
    args = parser.parse_args()
    output = args.output or Path("work") / datetime.now().strftime("run-%Y%m%d-%H%M%S-%f")
    if args.mode in ("physics", "coupled", "descent", "realvision", "scale"):
        if args.mode == "scale":
            from .scale_experiment import run_suite
        elif args.mode == "realvision":
            from .realvision import run_suite
        elif args.mode == "descent":
            from .descent_experiment import run_suite
        elif args.mode == "coupled":
            from .coupled_experiment import run_suite
        else:
            from .physics_experiment import run_suite
        run_suite(output)
        print(f"Replay: {(output / 'demo.html').resolve()}")
        return
    output.mkdir(parents=True, exist_ok=False)
    if args.mode == "delivery":
        from .visual_delivery_experiment import run_delivery_suite
        run_delivery_suite(output)
        print(f"Replay: {(output / 'demo.html').resolve()}")
        return
    if args.mode == "planning":
        from .localization_experiment import run_planning_suite
        run_planning_suite(output)
        print(f"Replay: {(output / 'demo.html').resolve()}")
        return
    if args.mode == "localization":
        from .localization_experiment import run_localization_suite
        run_localization_suite(output)
        print(f"Replay: {(output / 'demo.html').resolve()}")
        return
    if args.mode == "motion":
        from .motion_experiment import run_motion_suite
        report = run_motion_suite(output)
        for result in report["results"]:
            print(f"{result['key']}: {result['terminal_state']} | distance={result['distance_m']:.2f} m | time={result['elapsed_s']:.2f} s | violations={result['audit']['clearance_violations']}")
        print(f"Replay: {(output / 'demo.html').resolve()}")
        return
    if args.mode == "explore":
        from .exploration_experiment import run_exploration_suite
        report = run_exploration_suite(output)
        for result in report["results"]:
            print(f"{result['key']}: {result['terminal_state']} | moves={result['distance_m']} | violations={result['audit']['clearance_violations']}")
        print(f"Replay: {(output / 'demo.html').resolve()}")
        return
    if args.mode == "camera":
        from .camera_experiment import run_camera_suite
        report = run_camera_suite(output)
        for case in report["cases"]:
            m = case["metrics"]
            print(f"{case['key']}: points={m['depth_points']} | unknown_cells={m['unknown_cells']}")
        print(f"Replay: {(output / 'demo.html').resolve()}")
        return
    evaluation = None
    if args.mode == "vision":
        from .vision_experiment import run_vision_suite
        results, evaluation = run_vision_suite(output)
    else:
        results = [run_mission(scenario) for scenario in scenarios()]
    serialized = json.dumps(results, ensure_ascii=False, indent=2)
    (output / "results.json").write_text(serialized, encoding="utf-8")
    template = (Path(__file__).parent / "replay.html").read_text(encoding="utf-8")
    # 转义小于号，避免未来场景文本中的 HTML 标签结束 JSON script 元素。
    embedded = json.dumps(results, ensure_ascii=False).replace("<", "\\u003c")
    report = json.dumps(evaluation, ensure_ascii=False).replace("<", "\\u003c")
    (output / "demo.html").write_text(template.replace("__RESULTS_JSON__", embedded)
                                      .replace("__EVALUATION_JSON__", report), encoding="utf-8")
    for result in results:
        metric = result["metrics"]
        print(f"{result['scenario']}: {metric['terminal_state']} | moves={metric['grid_moves']} | plans={metric['planning_calls']}")
    print(f"Replay: {(output / 'demo.html').resolve()}")


if __name__ == "__main__":
    main()
