"""AI Shape-Sorting Robot Arm - Genesis 3D version (Franka Panda + top-down RGB-D camera).

Pipeline (same as the 2D version, now with real 3D physics & grasping):
  1. Perception : top-down RGB-D camera -> height map (depth) -> object blobs
  2. AI         : MLP classifies each blob silhouette (circle / square / triangle)
  3. Geometry   : pixel + depth -> 3D position, minAreaRect -> grasp yaw
  4. Planning   : Genesis IK + smooth joint interpolation
  5. Control    : joint PD position control, finger force control (physical grasp)
  6. Check      : finger opening after lift (proprioception) -> grasp success / retry

Run:
  python shape_ai.py                       # train classifier (~20 s, CPU)
  python genesis_sort.py                   # GPU
  python genesis_sort.py --cpu             # CPU (slow)
  python genesis_sort.py --vis             # also open the interactive viewer
Outputs: out/genesis_demo.mp4, out/results.json
"""
import argparse
import json
import os
import numpy as np
import cv2
import joblib
import genesis as gs
from shape_ai import classify, CLASSES

# ------------------------------------------------------------------ config
OBJ_H = 0.04                      # object height [m]
WORK_X, WORK_Y = (0.40, 0.68), (-0.22, 0.26)
BINS = {"circle": (0.38, -0.45), "square": (0.56, -0.45), "triangle": (0.74, -0.45)}
BIN_COLORS = {"circle": (0.85, 0.3, 0.3, 1), "square": (0.3, 0.75, 0.3, 1), "triangle": (0.3, 0.45, 0.9, 1)}
CAM_POS, CAM_FOV, CAM_RES = (0.55, 0.0, 1.30), 50, (640, 480)
HAND_DOWN_OFFSET = 0.11           # hand link height above object center when grasping (Genesis tutorial: 0.13 for 0.04 cube)
GRIP_FORCE = -1.5                 # finger force [N] (tune if objects slip: -0.5 ... -3.0)
REC_EVERY = 4                     # dt=0.01 -> 25 fps real time


def to_np(x):
    if hasattr(x, "cpu"):
        x = x.cpu().numpy()
    x = np.asarray(x)
    return x[0] if x.ndim == 4 else x  # drop env batch dim if present


def render(cam, depth=False):
    """Version-robust wrapper around cam.render()."""
    out = cam.render(rgb=True, depth=depth)
    rgb, dep = out[0], (out[1] if depth else None)
    rgb = to_np(rgb).astype(np.uint8)
    return (rgb, to_np(dep).astype(np.float32).squeeze()) if depth else rgb


def write_prism(path, side=0.05, h=OBJ_H):
    r = side / np.sqrt(3)
    v = [(r * np.cos(a), r * np.sin(a)) for a in np.deg2rad([90, 210, 330])]
    with open(path, "w") as f:
        for z in (-h / 2, h / 2):
            for x, y in v:
                f.write(f"v {x:.5f} {y:.5f} {z:.5f}\n")
        for face in ["1 3 2", "4 5 6", "1 2 5 4", "2 3 6 5", "3 1 4 6"]:
            f.write(f"f {face}\n")


def quat_down(yaw):
    """Gripper pointing down (Genesis wxyz), rotated by yaw about world z."""
    return np.array([0.0, np.cos(yaw / 2), np.sin(yaw / 2), 0.0])


def quat_z(yaw):
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


# ------------------------------------------------------------------ camera geometry
class TopCamera:
    """Pinhole model for the straight-down camera. The image->world axis convention
    is auto-calibrated once (8 candidate axis maps, matched to the initial scene)."""
    def __init__(self, cam):
        self.cam = cam
        W, H = CAM_RES
        self.cx, self.cy = W / 2, H / 2
        self.f = (H / 2) / np.tan(np.deg2rad(CAM_FOV) / 2)
        self.M = np.array([[0, -1], [-1, 0]], float)  # default guess (u,v) -> (x,y), refined by calibrate()

    def pix2world(self, u, v, depth):
        d = np.array([u - self.cx, v - self.cy]) * depth / self.f
        x, y = np.array(CAM_POS[:2]) + self.M @ d
        return np.array([x, y, CAM_POS[2] - depth])

    def calibrate(self, blobs, true_xy):
        best = None
        for swap in (False, True):
            for sx in (1, -1):
                for sy in (1, -1):
                    M = np.array([[0, sx], [sy, 0]], float) if swap else np.array([[sx, 0], [0, sy]], float)
                    self.M = M
                    est = np.array([self.pix2world(*b["uv"], b["depth"])[:2] for b in blobs])
                    err = np.mean([np.min(np.linalg.norm(true_xy - e, axis=1)) for e in est])
                    if best is None or err < best[0]:
                        best = (err, M)
        self.M = best[1]
        return best[0]


def perceive(topcam, clf):
    rgb, depth = render(topcam.cam, depth=True)
    height = CAM_POS[2] - depth
    mask = ((height > 0.015) & (height < 0.08)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, cents = cv2.connectedComponentsWithStats(mask)
    blobs = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 60:
            continue
        m = (lab == i).astype(np.uint8) * 255
        u, v = cents[i]
        d = float(np.median(depth[lab == i]))
        p = topcam.pix2world(u, v, d)
        cls, conf = classify(clf, m)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        (_, _), (_, _), ang = cv2.minAreaRect(cnts[0])
        blobs.append(dict(uv=(float(u), float(v)), depth=d, pos=p, cls=cls, conf=conf,
                          img_angle=float(ang), contour=cnts[0]))
    return rgb, blobs


def in_workspace(p):
    return WORK_X[0] - 0.03 < p[0] < WORK_X[1] + 0.03 and WORK_Y[0] - 0.03 < p[1] < WORK_Y[1] + 0.03


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--vis", action="store_true")
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    os.makedirs("out", exist_ok=True)
    os.makedirs("assets", exist_ok=True)
    write_prism("assets/tri_prism.obj")
    clf = joblib.load("shape_mlp_3d.joblib")

    gs.init(backend=gs.cpu if args.cpu else gs.gpu, seed=args.seed)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, substeps=4),
        viewer_options=gs.options.ViewerOptions(camera_pos=(1.6, -1.2, 1.2), camera_lookat=(0.5, -0.1, 0.1), camera_fov=40),
        show_viewer=args.vis,
    )
    scene.add_entity(gs.morphs.Plane())
    for c, (bx, by) in BINS.items():  # flat colored bin pads (2 mm, below detection threshold)
        scene.add_entity(gs.morphs.Box(size=(0.15, 0.15, 0.002), pos=(bx, by, 0.001), fixed=True),
                         surface=gs.surfaces.Default(color=BIN_COLORS[c]))
    obj_classes = ["circle", "square", "triangle"] * 2
    objs = []
    for i, c in enumerate(obj_classes):
        col = tuple(float(v) for v in rng.uniform(0.15, 0.85, 3)) + (1.0,)
        pos = (0.2, 0.6 + 0.1 * i, OBJ_H / 2)  # parked; placed per episode
        if c == "circle":
            m = gs.morphs.Cylinder(radius=0.022, height=OBJ_H, pos=pos)
        elif c == "square":
            m = gs.morphs.Box(size=(0.04, 0.04, OBJ_H), pos=pos)
        else:
            m = gs.morphs.Mesh(file=os.path.abspath("assets/tri_prism.obj"), pos=pos)
        objs.append(dict(cls=c, ent=scene.add_entity(m, surface=gs.surfaces.Default(color=col))))
    franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    top = scene.add_camera(res=CAM_RES, pos=CAM_POS, lookat=(CAM_POS[0], CAM_POS[1], 0.0), up=(1, 0, 0), fov=CAM_FOV, GUI=False)
    show = scene.add_camera(res=CAM_RES, pos=(1.45, -1.05, 0.95), lookat=(0.5, -0.12, 0.08), fov=40, GUI=False)
    scene.build()

    motors, fingers = np.arange(7), np.arange(7, 9)
    franka.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
    franka.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]))
    franka.set_dofs_force_range(np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
                                np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]))
    ee = franka.get_link("hand")
    topcam = TopCamera(top)

    writer = cv2.VideoWriter("out/genesis_raw.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 25, (CAM_RES[0] * 2, CAM_RES[1] + 60))
    state = dict(k=0, status="", blobs=[], target=None)

    def compose():
        a = cv2.cvtColor(render(show), cv2.COLOR_RGB2BGR)
        b = cv2.cvtColor(render(top), cv2.COLOR_RGB2BGR)
        for bl in state["blobs"]:
            hl = bl is state["target"]
            cv2.drawContours(b, [bl["contour"]], -1, (0, 0, 255) if hl else (0, 200, 255), 2)
            u, v = map(int, bl["uv"])
            cv2.putText(b, f'{bl["cls"]} {bl["conf"]:.2f}', (u - 40, v - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        cv2.putText(a, "Genesis 3D: Franka Panda", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(b, "Top RGB-D camera + AI (MLP)", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        bar = np.full((60, CAM_RES[0] * 2, 3), 40, np.uint8)
        cv2.putText(bar, state["status"], (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        writer.write(np.vstack([np.hstack([a, b]), bar]))

    def step(n=1):
        for _ in range(n):
            scene.step()
            state["k"] += 1
            if state["k"] % REC_EVERY == 0:
                compose()

    def move_to(pos, yaw, n=80, settle=25):
        q_goal = to_np(franka.inverse_kinematics(link=ee, pos=np.array(pos), quat=quat_down(yaw))).ravel()[:7]
        q0 = to_np(franka.get_dofs_position()).ravel()[:7]
        for k in range(n):
            s = (k + 1) / n
            s = 3 * s * s - 2 * s ** 3
            franka.control_dofs_position(q0 + (q_goal - q0) * s, motors)
            step()
        step(settle)

    def gripper(open_):
        if open_:
            franka.control_dofs_position(np.array([0.04, 0.04]), fingers)
        else:
            franka.control_dofs_force(np.array([GRIP_FORCE, GRIP_FORCE]), fingers)
        step(40)

    def place_objects():
        pts = []
        for o in objs:
            for _ in range(200):
                p = np.array([rng.uniform(*WORK_X), rng.uniform(*WORK_Y)])
                if all(np.linalg.norm(p - q) > 0.10 for q in pts):
                    break
            pts.append(p)
            o["ent"].set_pos(np.array([p[0], p[1], OBJ_H / 2 + 0.002]))
            o["ent"].set_quat(quat_z(rng.uniform(0, 2 * np.pi)))
        step(30)

    def true_xy():
        return np.array([to_np(o["ent"].get_pos()).ravel()[:2] for o in objs])

    home = np.array([0.45, 0.0, 0.45])
    log = []
    franka.control_dofs_position(np.array([0.04, 0.04]), fingers)
    state["status"] = "Start"
    move_to(home, 0.0, n=60)

    for ep in range(args.episodes):
        state["status"] = f"Episode {ep + 1}: 6 objects (random pose / color)"
        place_objects()
        if ep == 0:  # one-time camera axis calibration
            _, blobs = perceive(topcam, clf)
            err = topcam.calibrate(blobs, true_xy())
            print(f"[calib] mean position error after axis calibration: {err * 1000:.1f} mm")
        for attempt in range(12):
            _, blobs = perceive(topcam, clf)
            blobs = [b for b in blobs if in_workspace(b["pos"])]
            state["blobs"] = blobs
            if not blobs:
                break
            tgt = max(blobs, key=lambda b: b["conf"])
            state["target"] = tgt
            state["status"] = f"AI: {len(blobs)} objects -> target '{tgt['cls']}' (conf {tgt['conf']:.2f})"
            step(25)
            x, y, ztop = tgt["pos"]
            zc = ztop - OBJ_H / 2
            yaw = -np.deg2rad(tgt["img_angle"])  # sign handled by symmetry of square/triangle grasp
            state["status"] = f"Reach '{tgt['cls']}'"
            move_to((x, y, 0.25), yaw)
            move_to((x, y, zc + HAND_DOWN_OFFSET), yaw, n=50)
            state["status"] = "Grasp (finger force control)"
            gripper(open_=False)
            move_to((x, y, 0.28), yaw, n=50)
            width = float(np.sum(to_np(franka.get_dofs_position()).ravel()[7:9]))
            held = width > 0.006
            gt_i = int(np.argmin(np.linalg.norm(true_xy() - np.array([x, y]), axis=1)))
            rec = dict(episode=ep + 1, attempt=attempt + 1, pred=tgt["cls"], conf=round(tgt["conf"], 3),
                       true=objs[gt_i]["cls"], grasp_ok=held, finger_width_mm=round(width * 1000, 1))
            if not held:
                state["status"] = "Grasp failed (fingers closed) -> retry"
                gripper(open_=True)
                log.append(rec)
                continue
            bx, by = BINS[tgt["cls"]]
            state["status"] = f"Place into '{tgt['cls']}' bin"
            state["blobs"] = []
            move_to((bx, by, 0.28), yaw, n=90)
            move_to((bx, by, 0.18), yaw, n=40)
            gripper(open_=True)
            move_to((bx, by, 0.30), yaw, n=40)
            p = to_np(objs[gt_i]["ent"].get_pos()).ravel()
            rec["placed_in_correct_bin"] = bool(np.linalg.norm(p[:2] - np.array(BINS[rec["true"]])) < 0.09)
            log.append(rec)
        state["status"] = f"Episode {ep + 1} done"
        state["blobs"] = []
        move_to(home, 0.0, n=60)

    writer.release()
    picks = [r for r in log if r["grasp_ok"]]
    summary = dict(
        attempts=len(log), grasps_ok=len(picks),
        classification_acc=float(np.mean([r["pred"] == r["true"] for r in log])) if log else None,
        placed_correct=int(sum(r.get("placed_in_correct_bin", False) for r in picks)),
        total_objects=len(objs) * args.episodes, log=log)
    json.dump(summary, open("out/results.json", "w"), indent=1)
    print({k: v for k, v in summary.items() if k != "log"})


if __name__ == "__main__":
    main()
