# MIREIA Core

Physics and math utilities underlying the Driving Risk Field (DRF).

## Modules

| Module | Key Class | Description |
|--------|-----------|-------------|
| `physics.py` | `RiskOracle` | Main risk calculator: computes scalar risk at any world point from dynamic obstacles, static obstacles, and road boundaries |
| `constants.py` | — | Shared physical constants: gravity, friction coefficient, reaction time, Gaussian spread parameters for obstacle risk fields |

## Driving Risk Field (DRF)

The `RiskOracle` combines three additive risk components evaluated at the ego position:

### Dynamic obstacle risk

Each moving obstacle (vehicle, pedestrian) contributes a Gaussian risk field centred on its predicted position. The spread (σ) scales with speed and uncertainty. Risk falls off exponentially with distance.

```
r_dynamic(x) = Σ_i  A_i · exp(−‖x − p_i‖² / (2σ_i²))
```

### Static obstacle risk

Pre-baked risk from parked vehicles, barriers, and other immovable objects. Uses the same Gaussian kernel but with a fixed spread.

### Road boundary risk

Penalises proximity to lane boundaries and off-road regions. Computed from CARLA waypoint topology.

### Total risk

```
risk = clip(r_dynamic + r_static + r_boundary, 0, 1)
```

The `WorldManager` calls `RiskOracle.get_risk()` once per tick to produce the `ground_truth_risk` label written to `dataset.jsonl`.

## Constants

Key values from `constants.py`:

| Constant | Value | Meaning |
|----------|-------|---------|
| `GRAVITY` | 9.81 | m/s² |
| `FRICTION_COEFF` | 0.8 | Road–tyre friction (default CARLA asphalt) |
| `REACTION_TIME` | 1.0 | Driver reaction time (s) used in braking distance |
| `DRF_SIGMA_BASE` | (tuned) | Base Gaussian spread for static obstacles |
| `DRF_SIGMA_SPEED_SCALE` | (tuned) | Speed-dependent spread scaling for dynamic obstacles |
