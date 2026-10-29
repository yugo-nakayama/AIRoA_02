# AI Shape-Sorting Robot Arm（AI形状認識による仕分けロボットアーム）

Physical AI応用1講座 最終課題。
Camera image -> AI (MLP) shape recognition -> inverse kinematics + PD control -> pick & place into bins.

## Run
```bash
pip install -r requirements.txt
python vision.py   # generate synthetic data & train MLP (~15 s, CPU)
python main.py     # run 3 episodes, writes outputs/demo_raw.mp4 and outputs/results.json
ffmpeg -i outputs/demo_raw.mp4 -c:v libx264 -pix_fmt yuv420p outputs/demo.mp4
```
## Files
- `sim.py`    2-link planar arm (FK/IK, PD joint control), top-down camera renderer
- `vision.py` contour detection + MLP shape classifier (color/brightness-invariant silhouette feature)
- `main.py`   perception -> decision -> planning -> control loop, video recording

## Results
Classifier test accuracy 100% (900 images) / sorting 15/15 / mean reach error ~2.2 mm

## Genesis 3D version
See `genesis3d/README.md` (Franka Panda + top RGB-D camera, physical grasping).
