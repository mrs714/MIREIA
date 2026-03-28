from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import carla
import numpy as np

from MIREIA.config import Config
from MIREIA.data_collection.dataset_utils import load_jsonl_records
from MIREIA.simulation.scenarios import Scenario, get_default_ego_camera_position
from MIREIA.simulation.simple_route_controller import SimpleRouteController
from MIREIA.simulation.world_manager import WorldManager


@dataclass
class TrialDefinition:
    """Fixed world setup + fixed route used across multiple ego subtrials."""

    name: str
    route_start: tuple[float, float, float]
    route_end: tuple[float, float, float]
    description: str = ""
    map_name: str = "Town03"
    weather: str | dict = "ClearNoon"
    n_vehicles: int = 30
    n_pedestrians: int = 20
    pct_running: float = 0.0
    pct_crossing: float = 0.0
    safe_vehicles: bool = True
    seed: int = 42
    sync_mode: bool = True
    fixed_delta: float = 0.05

    @property
    def folder_path(self) -> str:
        return os.path.join(Config.PATH_TO_TRIALS, self.name)

    @property
    def json_path(self) -> str:
        return os.path.join(self.folder_path, "trial.json")

    @property
    def runs_path(self) -> str:
        return os.path.join(self.folder_path, "runs")

    def to_dict(self) -> dict:
        data = asdict(self)
        data["route_start"] = list(self.route_start)
        data["route_end"] = list(self.route_end)
        return data

    def save(self) -> None:
        os.makedirs(self.folder_path, exist_ok=True)
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, name: str) -> "TrialDefinition":
        json_path = os.path.join(Config.PATH_TO_TRIALS, name, "trial.json")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["route_start"] = tuple(data["route_start"])
        data["route_end"] = tuple(data["route_end"])
        return cls(**data)

    def create_subtrial_folder(self, subtrial_name: str) -> tuple[str, str]:
        os.makedirs(self.runs_path, exist_ok=True)
        run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{subtrial_name}"
        run_path = os.path.join(self.runs_path, run_id)
        os.makedirs(run_path, exist_ok=True)
        return run_id, run_path


@dataclass
class EgoTrialConfig:
    """Per-subtrial ego setup while world settings remain fixed."""

    name: str
    ego_blueprint: str = "vehicle.lincoln.mkz_2020"
    ego_spawn_index: int | None = None
    ego_camera_position: tuple[float, float, float] | None = None
    use_vehicle_camera_defaults: bool = True
    target_speed_kmh: float = 20.0
    speed_multiplier: float = 1.0
    controller_mode: str = "behavior_agent"
    controller_behavior: str = "normal"
    notes: str = ""


@dataclass
class TrialRunSummary:
    trial_name: str
    subtrial_name: str
    run_id: str
    run_path: str
    num_frames: int
    duration_s: float
    traveled_m: float
    risk_auc: float
    risk_distance_integral: float
    risk_per_meter: float
    predicted_risk_auc: float | None = None
    finished: bool = False
    metadata: dict = field(default_factory=dict)


class TrialRunner:
    """Runs fixed-world trials with different ego subtrial configs."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    @staticmethod
    def _compute_topdown_spectator_transform(wm: WorldManager,
                                             map_fov: float = 90.0,
                                             map_yaw: float = -90.0,
                                             map_roll: float = 0.0) -> carla.Transform | None:
        # Reuse WorldManager map bounds so spectator framing matches top-down camera framing.
        bounds = wm._WorldManager__compute_map_bounds()
        if bounds is None:
            return None

        center_x, center_y, size = bounds
        fov_rad = np.deg2rad(map_fov)
        height = (size / 2.0) / max(1e-6, np.tan(fov_rad / 2.0))

        return carla.Transform(
            carla.Location(x=center_x, y=-center_y, z=height),
            carla.Rotation(pitch=-90.0, yaw=map_yaw, roll=map_roll),
        )

    @classmethod
    def _set_spectator_like_topdown(cls, wm: WorldManager,
                                    map_fov: float = 90.0,
                                    map_yaw: float = -90.0,
                                    map_roll: float = 0.0) -> None:
        transform = cls._compute_topdown_spectator_transform(
            wm,
            map_fov=map_fov,
            map_yaw=map_yaw,
            map_roll=map_roll,
        )
        if transform is None or wm.world is None:
            return

        spectator = wm.world.get_spectator()
        if spectator is not None:
            spectator.set_transform(transform)

    @staticmethod
    def _draw_route_debug(world: carla.World,
                          controller: SimpleRouteController,
                          route_start: carla.Location,
                          route_end: carla.Location,
                          life_time: float = 0.08) -> None:
        world.debug.draw_point(
            location=carla.Location(x=route_start.x, y=route_start.y, z=route_start.z + 0.5),
            size=0.25,
            color=carla.Color(255, 0, 0),
            life_time=life_time,
        )
        world.debug.draw_point(
            location=carla.Location(x=route_end.x, y=route_end.y, z=route_end.z + 0.5),
            size=0.25,
            color=carla.Color(255, 0, 0),
            life_time=life_time,
        )
        controller.draw_plan(world, max_points=1400, life_time=life_time)

    @staticmethod
    def _should_draw_debug(step: int,
                           image_stride: int,
                           skip_after_capture_ticks: int = 1,
                           skip_before_capture_ticks: int = 0) -> bool:
        if image_stride <= 1:
            return False

        residue = step % image_stride
        if residue == 0:
            return False

        if skip_after_capture_ticks > 0 and residue <= skip_after_capture_ticks:
            return False

        if skip_before_capture_ticks > 0 and residue >= image_stride - skip_before_capture_ticks:
            return False

        return True

    def run_subtrial(
        self,
        trial: TrialDefinition,
        ego_cfg: EgoTrialConfig,
        max_steps: int = 10000,
        image_stride: int = 1,
        store_topdown_images: bool = False,
        store_risk_frame_images: bool = False,
        store_static_risk_map: bool = False,
        draw_debug_every_tick: bool = True,
        draw_debug_skip_after_capture_ticks: int = 1,
        draw_debug_skip_before_capture_ticks: int = 0,
        predictor_fn: Callable[[dict], float] | None = None,
    ) -> TrialRunSummary:
        image_stride = max(1, int(image_stride))
        trial.save()
        run_id, run_path = trial.create_subtrial_folder(ego_cfg.name)
        run_path_p = Path(run_path)
        images_dir = run_path_p / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        controller = SimpleRouteController(
            target_speed=ego_cfg.target_speed_kmh,
            sampling_resolution=2.0,
            mode=ego_cfg.controller_mode,
            behavior=ego_cfg.controller_behavior,
        )

        if ego_cfg.speed_multiplier != 1.0:
            controller.set_speed_modifier(lambda base_speed, tick_idx, _c: ego_cfg.speed_multiplier * base_speed)

        wm = WorldManager(sync_mode=trial.sync_mode, fixed_delta=trial.fixed_delta, verbose=self.verbose)
        wm.set_ego_controller(controller)

        camera_position = ego_cfg.ego_camera_position
        if camera_position is None and ego_cfg.use_vehicle_camera_defaults:
            camera_position = get_default_ego_camera_position(ego_cfg.ego_blueprint)

        scenario = Scenario(
            name=f"trial_{trial.name}__{ego_cfg.name}",
            map_name=trial.map_name,
            description=trial.description,
            weather=trial.weather,
            ego_blueprint=ego_cfg.ego_blueprint,
            ego_camera_position=camera_position,
            ego_spawn_index=ego_cfg.ego_spawn_index,
            ego_spawn_point=trial.route_start,
            ego_autopilot=False,
            n_vehicles=trial.n_vehicles,
            n_pedestrians=trial.n_pedestrians,
            pct_running=trial.pct_running,
            pct_crossing=trial.pct_crossing,
            safe_vehicles=trial.safe_vehicles,
            seed=trial.seed,
        )
        try:
            wm.load_scenario(scenario)
            wm.setup_sensors(save_dir=str(images_dir), enable_map_camera=store_topdown_images)
            self._set_spectator_like_topdown(wm)
            wm.enable_recording(
                append=False,
                include_topdown=store_topdown_images,
                include_static_risk_image=False,
                jsonl_path=str(run_path_p / "dataset.jsonl"),
                static_meta={
                    "trial_name": trial.name,
                    "subtrial_name": ego_cfg.name,
                    "run_id": run_id,
                    "route_start": list(trial.route_start),
                    "route_end": list(trial.route_end),
                    "ego_config": asdict(ego_cfg),
                },
            )

            rel_static_risk = ""
            if store_static_risk_map:
                static_risk_path = images_dir / "risk_static.png"
                wm.save_static_risk_map_image(save_path=str(static_risk_path))
                rel_static_risk = str(static_risk_path.relative_to(run_path_p))

            start_loc = wm.ego_vehicle.get_location()
            end_loc = carla.Location(*trial.route_end)
            controller.set_destination(start_loc, end_loc)

            started = time.time()
            finished = False
            route_start = carla.Location(*trial.route_start)

            for step in range(max_steps):
                rgb_path = images_dir / f"rgb_{step:06d}.png"
                rel_rgb = str(rgb_path.relative_to(run_path_p))
                topdown_path = images_dir / f"topdown_{step:06d}.png"
                rel_topdown = str(topdown_path.relative_to(run_path_p)) if store_topdown_images else ""
                risk_frame_path = images_dir / f"risk_{step:06d}.png"
                rel_risk_frame = str(risk_frame_path.relative_to(run_path_p)) if store_risk_frame_images else ""
                rel_risk = rel_risk_frame if store_risk_frame_images else rel_static_risk

                if draw_debug_every_tick and self._should_draw_debug(
                    step=step,
                    image_stride=image_stride,
                    skip_after_capture_ticks=draw_debug_skip_after_capture_ticks,
                    skip_before_capture_ticks=draw_debug_skip_before_capture_ticks,
                ):
                    self._draw_route_debug(
                        world=wm.world,
                        controller=controller,
                        route_start=route_start,
                        route_end=end_loc,
                        life_time=0.08,
                    )

                def _tick_and_log() -> None:
                    wm.tick(
                        ground_truth_risk=None,
                        rgb_image_path=rel_rgb,
                        topdown_image_path=rel_topdown,
                        risk_map_image_path=rel_risk,
                        extra_fields={
                            "trial_step": step,
                            "target_speed_kmh": controller.get_last_applied_target_speed(),
                        },
                    )

                def _tick_without_log() -> None:
                    if wm.traffic_handler is not None:
                        wm.traffic_handler.run_ego_controller_step()
                    wm.world.tick()
                    if wm.bridge is not None:
                        wm.bridge.update()

                if step % image_stride == 0:
                    wm.sensor_manager.save_ego_frame(save_path=str(rgb_path), tick_fn=_tick_and_log)
                    if store_topdown_images:
                        wm.sensor_manager.save_map_frame(
                            save_path=str(topdown_path),
                            tick_fn=_tick_without_log,
                        )
                    if store_risk_frame_images:
                        wm.save_risk_frame_image(save_path=str(risk_frame_path))
                else:
                    _tick_without_log()

                if controller.done():
                    finished = True
                    break

            elapsed = time.time() - started

            jsonl_path = run_path_p / "dataset.jsonl"
            records = load_jsonl_records(str(jsonl_path))

            if predictor_fn is not None:
                self._append_predicted_risk(jsonl_path, records, predictor_fn)
                records = load_jsonl_records(str(jsonl_path))

            summary = self._build_summary(
                trial=trial,
                ego_cfg=ego_cfg,
                run_id=run_id,
                run_path=run_path,
                records=records,
                finished=finished,
                elapsed=elapsed,
                sample_dt=trial.fixed_delta * image_stride,
            )

            with open(run_path_p / "summary.json", "w", encoding="utf-8") as f:
                json.dump(asdict(summary), f, indent=2)

            return summary
        finally:
            wm.destroy()

    def run_trial(
        self,
        trial: TrialDefinition,
        ego_configs: list[EgoTrialConfig],
        max_steps: int = 10000,
        image_stride: int = 1,
        store_topdown_images: bool = False,
        store_risk_frame_images: bool = False,
        store_static_risk_map: bool = False,
        draw_debug_every_tick: bool = True,
        draw_debug_skip_after_capture_ticks: int = 1,
        draw_debug_skip_before_capture_ticks: int = 0,
        predictor_fn: Callable[[dict], float] | None = None,
    ) -> list[TrialRunSummary]:
        summaries: list[TrialRunSummary] = []
        for ego_cfg in ego_configs:
            summaries.append(
                self.run_subtrial(
                    trial=trial,
                    ego_cfg=ego_cfg,
                    max_steps=max_steps,
                    image_stride=image_stride,
                    store_topdown_images=store_topdown_images,
                    store_risk_frame_images=store_risk_frame_images,
                    store_static_risk_map=store_static_risk_map,
                    draw_debug_every_tick=draw_debug_every_tick,
                    draw_debug_skip_after_capture_ticks=draw_debug_skip_after_capture_ticks,
                    draw_debug_skip_before_capture_ticks=draw_debug_skip_before_capture_ticks,
                    predictor_fn=predictor_fn,
                )
            )
        return summaries

    @staticmethod
    def _append_predicted_risk(
        jsonl_path: Path,
        records: list[dict],
        predictor_fn: Callable[[dict], float],
    ) -> None:
        for rec in records:
            try:
                rec["predicted_risk"] = float(predictor_fn(rec))
            except Exception:
                rec["predicted_risk"] = None

        with open(jsonl_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    @staticmethod
    def _build_summary(
        trial: TrialDefinition,
        ego_cfg: EgoTrialConfig,
        run_id: str,
        run_path: str,
        records: list[dict],
        finished: bool,
        elapsed: float,
        sample_dt: float | None = None,
    ) -> TrialRunSummary:
        dt = float(sample_dt) if sample_dt is not None else trial.fixed_delta
        gt = np.array([float(r.get("ground_truth_risk", 0.0)) for r in records], dtype=np.float64)
        risk_auc = float(np.trapz(gt, dx=dt)) if gt.size > 0 else 0.0

        positions = [
            (
                float(r.get("ego", {}).get("position", {}).get("x", 0.0)),
                float(r.get("ego", {}).get("position", {}).get("y", 0.0)),
            )
            for r in records
        ]
        if len(positions) >= 2:
            pos_arr = np.array(positions, dtype=np.float64)
            seg = np.linalg.norm(np.diff(pos_arr, axis=0), axis=1)
            traveled_m = float(seg.sum())
        else:
            seg = np.array([], dtype=np.float64)
            traveled_m = 0.0

        if seg.size > 0 and gt.size >= 2:
            n = min(seg.size, gt.size - 1)
            risk_distance_integral = float(np.sum(0.5 * (gt[:n] + gt[1:n + 1]) * seg[:n]))
        else:
            risk_distance_integral = 0.0

        risk_per_meter = risk_distance_integral / traveled_m if traveled_m > 0 else 0.0

        pred_vals = [r.get("predicted_risk") for r in records if r.get("predicted_risk") is not None]
        pred_auc = None
        if pred_vals:
            pred_arr = np.array([float(v) for v in pred_vals], dtype=np.float64)
            pred_auc = float(np.trapz(pred_arr, dx=dt))

        return TrialRunSummary(
            trial_name=trial.name,
            subtrial_name=ego_cfg.name,
            run_id=run_id,
            run_path=run_path,
            num_frames=len(records),
            duration_s=float(elapsed),
            traveled_m=traveled_m,
            risk_auc=risk_auc,
            risk_distance_integral=risk_distance_integral,
            risk_per_meter=risk_per_meter,
            predicted_risk_auc=pred_auc,
            finished=finished,
            metadata={
                "map_name": trial.map_name,
                "weather": trial.weather,
                "route_start": list(trial.route_start),
                "route_end": list(trial.route_end),
            },
        )
