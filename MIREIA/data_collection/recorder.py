"""
DatasetLogger — append-only JSONL writer for simulation frame data.

Each call to :meth:`log_frame` serialises one JSON object and appends it as a
single line to the output file.  Because every line is a self-contained JSON
document the dataset is never in a corrupt state: if CARLA crashes mid-run we
keep all frames written so far.

Typical usage (managed by WorldManager)::

    logger = DatasetLogger("scenarios/sunny_light_traffic/dataset.jsonl")
    for step in range(N):
        wm.tick()
        logger.log_frame(bridge=wm.bridge,
                 scenario=wm.scenario,
                 ego_vehicle=wm.ego_vehicle,
                 frame_id=step,
                 ground_truth_risk=risk_value,
                 rgb_image_path=f"images/rgb_{step:06d}.png",
                 topdown_image_path=f"images/topdown_{step:06d}.png",
                 risk_map_image_path="images/risk_map.png")
    logger.close()
"""

from __future__ import annotations

import json
import os
import time
from typing import TextIO

import carla

from MIREIA.simulation.bridge import (
    SimulationBridge,
    EgoKinematics,
    EnvironmentState,
)
from MIREIA.core.physics import RiskOracle
from MIREIA.analysis.plotter import RiskGrid


class DatasetLogger:
    """
    Append-only JSONL writer that records one simulation frame per line.

    The file is flushed after every frame so that data is preserved even if
    the simulation crashes.

    Parameters
    ----------
    output_path : str
        Path to the ``.jsonl`` file.  Parent directories are created
        automatically.
    append : bool
        If *True* (default) new frames are appended to an existing file;
        if *False* the file is truncated on open.
    """

    def __init__(self, output_path: str, append: bool = True, delete_existing: bool = False,
                 static_meta: dict | None = None):
        self.output_path = output_path
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        if delete_existing and os.path.exists(output_path):
            if os.path.isfile(output_path):
                os.remove(output_path)
        
        mode = "a" if append else "w"
        self._file: TextIO = open(output_path, mode, encoding="utf-8")
        self._frame_count: int = 0
        self._static_meta: dict = dict(static_meta or {})

    # ── Public API ──────────────────────────────────────────────────
    def log_frame(
        self,
        bridge: SimulationBridge,
        scenario,                        # Scenario (import-free to avoid circular deps)
        ego_vehicle: carla.Actor,
        frame_id: int,
        ground_truth_risk: float | None,
        rgb_image_path: str = "",
        topdown_image_path: str = "",
        risk_map_image_path: str = "",
        timestamp: float | None = None,
        risk_oracle: RiskOracle | None = None,
        baked_static_risk: RiskGrid | None = None,
        extra_fields: dict | None = None,
    ) -> dict:
        """
        Build a frame record from live simulation state and write it as a
        single JSON line.

        Parameters
        ----------
        bridge : SimulationBridge
            Provides ego kinematics, obstacles, pedestrians and env state.
        scenario : Scenario
            Provides static environment metadata (map, weather, seed ...).
        ego_vehicle : carla.Actor
            The ego CARLA actor (used to read vehicle controls).
        frame_id : int
            Sequential frame counter managed by the caller.
        ground_truth_risk : float | None
            Ground-truth risk label for this frame. If None or 0.0, a risk
            value is computed with the risk oracle.
        rgb_image_path : str
            Relative path to the saved front-camera RGB image.
        topdown_image_path : str
            Relative path to the saved top-down RGB image.
        risk_map_image_path : str
            Relative path to the saved static risk map image.
        timestamp : float | None
            Wall-clock timestamp.  Defaults to ``time.time()``.

        Returns
        -------
        dict
            The record that was written (useful for debugging / unit tests).
        """
        if timestamp is None:
            timestamp = time.time()

        if ground_truth_risk is None or ground_truth_risk == 0.0:
            if risk_oracle is None or baked_static_risk is None:
                raise ValueError(
                    "ground_truth_risk is None/0.0 but risk_oracle or baked_static_risk is missing"
                )
            ego = bridge.get_ego_kinematics()
            ground_truth_risk = risk_oracle.calculate_scene_risk(
                (ego.x, ego.y), bridge, baked_static_risk
            )

        record = self._build_record(
            bridge, scenario, ego_vehicle,
            frame_id, ground_truth_risk,
            rgb_image_path,
            topdown_image_path,
            risk_map_image_path,
            timestamp,
            extra_fields=extra_fields,
        )

        self._write_line(record)
        self._frame_count += 1
        return record

    @property
    def frame_count(self) -> int:
        """Number of frames written so far."""
        return self._frame_count

    def flush(self):
        """Force-flush the underlying file buffer."""
        if self._file and not self._file.closed:
            self._file.flush()

    def close(self):
        """Flush and close the JSONL file."""
        if self._file and not self._file.closed:
            self._file.flush()
            self._file.close()

    # ── Context-manager support ─────────────────────────────────────
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    # ── Internal helpers ────────────────────────────────────────────
    def _write_line(self, record: dict):
        line = json.dumps(record, ensure_ascii=False)
        self._file.write(line + "\n")
        self._file.flush()          # crash-safe: every frame hits disk

    def _build_record(
        self,
        bridge: SimulationBridge,
        scenario,
        ego_vehicle: carla.Actor,
        frame_id: int,
        ground_truth_risk: float,
        rgb_image_path: str,
        topdown_image_path: str,
        risk_map_image_path: str,
        timestamp: float,
        extra_fields: dict | None = None,
    ) -> dict:
        ego = bridge.get_ego_kinematics()
        env = bridge.get_environment_state()

        record = {
            # ── Metadata ────────────────────────────────────────────
            "frame_id":           frame_id,
            "timestamp":          timestamp,

            # ── Image path (relative) ───────────────────────────────
            "rgb_image_path":     rgb_image_path,
            "topdown_image_path": topdown_image_path,
            "risk_map_image_path": risk_map_image_path,

            # ── Ground-truth targets ────────────────────────────────
            "ground_truth_risk":  ground_truth_risk,
            "true_ego_speed":     ego.v,

            # ── Ego vehicle physics ─────────────────────────────────
            "ego": self._serialize_ego(ego, ego_vehicle),

            # ── Environment ─────────────────────────────────────────
            "environment": self._serialize_environment(env, scenario),
        }

        if self._static_meta:
            record["meta"] = dict(self._static_meta)
        if extra_fields:
            extra = dict(extra_fields)
            for key in (
                "predicted_risk",
                "predicted_risk_window",
                "predicted_risk_ready",
                "predicted_risk_buffer_size",
            ):
                if key in extra:
                    record[key] = extra[key]
            record["extra_fields"] = extra

        return record

    # ── Ego ─────────────────────────────────────────────────────────
    @staticmethod
    def _serialize_ego(ego: EgoKinematics, actor: carla.Actor) -> dict:
        ctrl = actor.get_control()
        return {
            "position": {"x": ego.x, "y": ego.y,
                         "z": actor.get_transform().location.z},
            "heading":   ego.heading,
            "speed":     ego.v,
            "velocity":  {"vx": ego.vx, "vy": ego.vy},
            "controls": {
                "throttle": ctrl.throttle,
                "steer":    ctrl.steer,
                "brake":    ctrl.brake,
            },
        }

    # ── Environment ─────────────────────────────────────────────────
    @staticmethod
    def _serialize_environment(env: EnvironmentState, scenario) -> dict:
        return {
            "map_name":    scenario.map_name,
            "weather":     scenario.weather,
            "seed":        scenario.seed,
            "visibility":  env.visibility,
            "friction":    env.mu,
        }

    # ── Repr ────────────────────────────────────────────────────────
    def __repr__(self):
        status = "open" if (self._file and not self._file.closed) else "closed"
        return (f"DatasetLogger(path='{self.output_path}', "
                f"frames={self._frame_count}, status={status})")