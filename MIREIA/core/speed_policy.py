"""Speed policies for risk-aware vehicle control.

Pure-math module (no CARLA, no torch) — safe to import anywhere.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class KineticRiskSpeedPolicy:
    """Risk-aware speed policy based on kinetic-energy scaling.

    Computes a safe target speed from the current zone speed limit and a
    scene-risk value using the formula:

        v_safe = sqrt( v_min² + (v_target² − v_min²) · exp(−λ · max(0, R − R_base)) )

    Properties
    ----------
    r_base : float
        Risk threshold below which the exponent is 0, so v_safe == v_target
        (no slowdown).  Default 3.0 matches the typical MIREIA GT-risk range.
    lam : float
        Response aggressiveness λ.  Higher → steeper speed drop above r_base.
    v_min_factor : float
        Speed floor as a fraction of v_target: v_min = v_min_factor * v_target.
        Guarantees the car never fully stops on a high-speed road.

    Usage
    -----
    >>> policy = KineticRiskSpeedPolicy(lam=0.3, v_min_factor=0.2)
    >>> policy(50.0, 5.0)   # v_target=50 km/h, risk=5 (above baseline)
    >>> # use as risk_speed_fn directly
    >>> runner.run_subtrial(..., risk_speed_fn=policy, use_ground_truth_risk=True)
    """

    r_base: float = 3.0
    lam: float = 0.3
    v_min_factor: float = 0.2

    def __call__(self, v_target: float, risk: float | None) -> float:
        """Return v_safe (km/h) for the given zone speed limit and risk value."""
        if risk is None:
            return float(v_target)
        v_min = self.v_min_factor * v_target
        excess = max(0.0, float(risk) - self.r_base)
        decay = math.exp(-self.lam * excess)
        return math.sqrt(v_min ** 2 + (v_target ** 2 - v_min ** 2) * decay)

    def preview(
        self,
        v_values: tuple[float, ...] = (30.0, 50.0, 90.0),
        r_values: tuple[float | None, ...] | None = None,
    ) -> str:
        """Return a formatted table string for sanity-checking the policy shape."""
        if r_values is None:
            r_values = (
                None,
                0.0,
                self.r_base - 1.0,
                self.r_base,
                self.r_base + 1.0,
                self.r_base + 3.0,
                self.r_base + 7.0,
                self.r_base + 17.0,
            )
        lines = [
            f"KineticRiskSpeedPolicy  r_base={self.r_base}  lam={self.lam}  "
            f"v_min_factor={self.v_min_factor}",
            f"  {'v_target':>9s} {'R':>6s} -> {'v_safe':>8s}  {'ratio':>6s}",
        ]
        for v in v_values:
            for r in r_values:
                out = self(v, r)
                r_disp = "None" if r is None else f"{r:.1f}"
                ratio = f"{out / v:.3f}" if v > 0 else "  n/a"
                lines.append(f"  {v:9.1f} {r_disp:>6s} -> {out:8.2f}  {ratio:>6s}")
            lines.append("")
        return "\n".join(lines)
