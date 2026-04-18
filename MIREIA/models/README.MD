MIREIA models directory.

Common expected files:

- raft/ (RAFT checkpoints, typically .pth)
- e2e_risk_checkpoint.pt
- bdu_gru_risk_checkpoint_stable.pt
- environment_multitask_checkpoint.pt
- road_segmentation_multitask_checkpoint.pt
- road_segmentation_multitask_best.pt
- yolo11s.pt
- depth_anything_v2_vits.pth

Training split controls (used in train_demo.ipynb):

- partition_mode:
	- "scenario": validation is selected by scenario-name token(s).
	- "frame": all scenarios are used, with train/val split by frame order.
- val_scenario_tokens: list of scenario name tokens for validation when partition_mode="scenario".
	- Example: ["Town10HD"] (default behavior)
	- Example: ["Town01"]
- frame_train_ratio: train fraction when partition_mode="frame".
	- Accepts 0.7 or 70 for a 70/30 split.

Examples:

- Scenario-token validation:
	- partition_mode="scenario"
	- val_scenario_tokens=["Town01"]
- Frame-based split over all scenarios:
	- partition_mode="frame"
	- frame_train_ratio=0.7
