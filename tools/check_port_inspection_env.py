from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mathbased_mcpp.port_inspection import PortInspectionSchedulingEnv, load_inspection_tasks, load_port_grid
from mathbased_mcpp.port_inspection.platform_params import load_platform_profiles, platform_from_profile
from mathbased_mcpp.port_inspection.simple_planner import create_platforms
from mathbased_mcpp.port_inspection.v12_contract import classify_config_boundary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Smoke-check the port inspection scheduling environment.")
    parser.add_argument("--config", default="configs/port_yangshan_task_initial_v1.toml")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    config = _load_config(args.config)
    env = build_env(config)
    rng = np.random.default_rng(args.seed)
    observation = env.reset(seed=args.seed)
    print(f"contract_boundary={json.dumps(env.contract_boundary, ensure_ascii=False)}")
    print(f"observation_dim={observation.shape[0]}")
    print(f"action_dim={env.action_dim}")
    for step in range(args.steps):
        mask = env.action_masks()
        action = [int(rng.choice(np.flatnonzero(row))) for row in mask]
        result = env.step(action)
        print(
            json.dumps(
                {
                    "step": step + 1,
                    "action": action,
                    "reward": result.reward,
                    "reward_terms": result.info["reward_terms"],
                    "late_tasks": result.info["late_tasks"],
                    "completed_count": len(result.info["completed_tasks"]),
                    "valid_action_count": int(mask.sum()),
                },
                ensure_ascii=False,
            )
        )
        if result.done:
            break


def build_env(config: dict[str, object]) -> PortInspectionSchedulingEnv:
    grid = load_port_grid(str(config["grid_path"]))
    tasks = load_inspection_tasks(str(config["tasks_path"]), grid)
    platforms = _platforms_from_config(config, grid.depot)
    rl_config = dict(config.get("scheduler_rl", {}))
    reward_weights = dict(rl_config.get("reward", {}))
    scheduling_config = dict(config.get("scheduling", {}))
    env = PortInspectionSchedulingEnv(
        grid=grid,
        tasks=tasks,
        platforms=platforms,
        max_steps=int(rl_config.get("max_steps", 64)),
        candidate_k=int(rl_config.get("candidate_k", config.get("candidate_k", 8))),
        reward_weights={key: float(value) for key, value in reward_weights.items()},
        candidate_weights={key: float(value) for key, value in scheduling_config.items() if isinstance(value, (int, float))},
        review_trigger=dict(config.get("review_trigger", {})),
    )
    env.contract_boundary = classify_config_boundary(config).as_dict()
    return env


def _platforms_from_config(config: dict[str, object], depot: tuple[int, int]):
    profiles_path = config.get("platform_profiles_path")
    sequence = [str(item) for item in config.get("platform_profile_sequence", [])]  # type: ignore[arg-type]
    if profiles_path and sequence:
        profiles = load_platform_profiles(str(profiles_path))
        counts: dict[str, int] = {}
        platforms = []
        for profile_name in sequence:
            profile = profiles[profile_name]
            prefix = str(profile.get("platform_type", profile_name.split("_", 1)[0])).upper()
            counts[prefix] = counts.get(prefix, 0) + 1
            platforms.append(platform_from_profile(f"{prefix}-{counts[prefix]}", profile_name, profile, depot))
        _attach_platform_depots(platforms, config)
        return platforms

    platform_config = dict(config.get("platform", {}))
    platforms = create_platforms(
        depot=depot,
        uav_count=int(config.get("uav_count", 1)),
        usv_count=int(config.get("usv_count", 1)),
        uav_config=dict(platform_config.get("uav", {})),
        usv_config=dict(platform_config.get("usv", {})),
    )
    _attach_platform_depots(platforms, config)
    return platforms


def _attach_platform_depots(platforms, config: dict[str, object]) -> None:
    platform_depots = dict(config.get("platform_depots", {}))
    for platform in platforms:
        depot = platform_depots.get(platform.platform_type.lower()) or platform_depots.get(platform.platform_type.upper())
        if isinstance(depot, (list, tuple)) and len(depot) == 2:
            platform.current_cell = (int(depot[0]), int(depot[1]))
            platform.route = [platform.current_cell]
            platform.metadata["depot_cell"] = list(platform.current_cell)


def _load_config(path: str | Path) -> dict[str, object]:
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


if __name__ == "__main__":
    main()
