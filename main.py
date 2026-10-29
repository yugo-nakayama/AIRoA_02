"""AI sorting robot: camera -> MLP shape recognition -> IK + PD control -> pick & place into bins."""
import numpy as np
import cv2
import joblib
import json
from sim import Arm, ik, fk, render_scene, random_objects, BINS, W, H
from vision import detect

FPS, SUBSTEPS = 30, 4
rng = np.random.default_rng(7)
clf = joblib.load("data/shape_mlp.joblib")
writer = cv2.VideoWriter("outputs/demo_raw.mp4", cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W * 2, H + 60))
log = []
binned = []


def frame(objs, arm, held, status, dets=None):
    sim_img = render_scene(objs, arm, held, noise=False, binned=binned)
    cam = render_scene(objs, None, None, noise=True, rng=rng, binned=binned)  # what the AI sees
    for d in dets or []:
        x, y = map(int, d["px"])
        cv2.rectangle(cam, (x - 26, y - 26), (x + 26, y + 26), (0, 0, 255), 2)
        cv2.putText(cam, f'{d["cls"]} {d["conf"]:.2f}', (x - 30, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 200), 1)
    cv2.putText(sim_img, "Robot (simulation)", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(cam, "Camera + AI inference (MLP)", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    bar = np.full((60, W * 2, 3), 40, np.uint8)
    cv2.putText(bar, status, (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    writer.write(np.vstack([np.hstack([sim_img, cam]), bar]))


def move(arm, target, objs, held, status, dets, secs=1.2):
    q0, q1 = arm.q.copy(), ik(target)
    n = int(secs * FPS)
    for k in range(n):
        s = (k + 1) / n
        s = 3 * s**2 - 2 * s**3  # smooth (cubic) interpolation
        for _ in range(SUBSTEPS):
            arm.step(q0 + (q1 - q0) * s)
        frame(objs, arm, held, status, dets)
    for _ in range(int(0.3 * FPS)):  # settle
        for _ in range(SUBSTEPS):
            arm.step(q1)
        frame(objs, arm, held, status, dets)
    return np.linalg.norm(fk(arm.q)[1] - target)


def hold(n, *a):
    for _ in range(int(n * FPS)):
        frame(*a)


arm = Arm()
HOME = np.array([0.0, 0.35])
for episode in range(3):
    objs = random_objects(5, rng)
    hold(1.5, objs, arm, None, f"Episode {episode+1}: new objects placed (random color / size / rotation)")
    while True:
        cam = render_scene(objs, None, None, noise=True, rng=rng, binned=binned)
        dets = detect(cam, clf)
        if not dets:
            break
        hold(1.0, objs, arm, None, f"Perception: {len(dets)} objects detected & classified by AI", dets)
        d = min(dets, key=lambda d: np.linalg.norm(d["pos"] - fk(arm.q)[1]))  # nearest first
        obj = min(objs, key=lambda o: np.linalg.norm(o["pos"] - d["pos"]))
        err = move(arm, d["pos"], objs, None, f"Reach -> {d['cls']} (conf {d['conf']:.2f})", [d])
        held = obj if err < 0.03 else None
        hold(0.3, objs, arm, held, "Grasp" if held else "Grasp FAILED", [d])
        if held is None:
            break
        move(arm, BINS[d["cls"]], objs, held, f"Place into '{d['cls']}' bin", [])
        objs = [o for o in objs if o is not obj]
        obj["pos"] = fk(arm.q)[1] + rng.uniform(-0.03, 0.03, 2)
        binned.append(obj)
        log.append(dict(episode=episode + 1, true=str(obj["cls"]), pred=d["cls"], conf=round(d["conf"], 3),
                        reach_err_mm=round(err * 1000, 1)))
        hold(0.2, objs, arm, None, "Release", [])
    move(arm, HOME, objs, None, "Done -> home", [])

writer.release()
acc = np.mean([l["true"] == l["pred"] for l in log])
summary = dict(n_picks=len(log), sort_accuracy=float(acc),
               mean_reach_err_mm=float(np.mean([l["reach_err_mm"] for l in log])), log=log)
json.dump(summary, open("outputs/results.json", "w"), indent=1)
print({k: v for k, v in summary.items() if k != "log"})
