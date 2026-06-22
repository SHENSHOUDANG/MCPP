from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

from .schema import GridCell, Platform


def load_platform_profiles(path: str | Path) -> dict[str, dict[str, Any]]:
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)
    profiles = raw.get("profiles", raw)
    return {str(name): dict(value) for name, value in profiles.items()}


def platform_from_profile(platform_id: str, profile_name: str, profile: dict[str, Any], depot: GridCell) -> Platform:
    platform_type = str(profile.get("platform_type", "UAV" if profile_name.upper().startswith("UAV") else "USV")).upper()
    return Platform(
        platform_id=platform_id,
        platform_type=platform_type,
        current_cell=depot,
        speed_mps=float(profile["speed_mps"]),
        endurance_minutes=float(profile.get("endurance_min", profile.get("endurance_minutes", 30.0))),
        allowed_task_types=tuple(str(item) for item in profile.get("allowed_task_types", ("point", "line", "area"))),
        preferred_task_types=tuple(str(item) for item in profile.get("preferred_task_types", ())),
        max_speed_mps=float(profile.get("max_speed_mps", 0.0)),
        nominal_endurance_minutes=float(profile.get("nominal_endurance_min", profile.get("nominal_endurance_minutes", 0.0))),
        return_reserve_ratio=float(profile.get("return_reserve_ratio", 0.15)),
        sensor_radius_m=float(profile.get("sensor_radius_m", 0.0)),
        energy_capacity=float(profile.get("energy_capacity", 1.0)),
        energy=float(profile.get("energy_capacity", 1.0)),
        payload_kg=float(profile.get("payload_kg", 0.0)),
        coverage_width_cells=int(profile.get("coverage_width_cells", 1)),
        energy_rate_per_cell=float(profile.get("energy_rate", profile.get("energy_rate_per_cell", 1.0))),
        metadata={"profile": profile_name, **{key: value for key, value in profile.items() if key not in {"allowed_task_types", "preferred_task_types"}}},
        route=[depot],
    )
