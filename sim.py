"""2-link planar arm simulator + top-down camera renderer (numpy/OpenCV only)."""
import numpy as np
import cv2

L1, L2 = 0.30, 0.25          # link lengths [m]
BASE = np.array([0.0, 0.0])  # base position (world)
SCALE = 520                  # px per m
W, H = 640, 480              # camera image size
ORIGIN = np.array([W // 2, H - 30])  # pixel of world (0,0)

CLASSES = ["circle", "square", "triangle"]
# drop bins (world coords) per class
BINS = {"circle": np.array([-0.40, 0.08]), "square": np.array([0.40, 0.08]), "triangle": np.array([0.0, 0.50])}


def w2p(p):
    return (ORIGIN + np.array([p[0], -p[1]]) * SCALE).astype(int)


def p2w(px):
    d = (np.array(px, float) - ORIGIN) / SCALE
    return np.array([d[0], -d[1]])


def fk(q):
    e = BASE + L1 * np.array([np.cos(q[0]), np.sin(q[0])])
    t = e + L2 * np.array([np.cos(q[0] + q[1]), np.sin(q[0] + q[1])])
    return e, t


def ik(p, elbow_up=True):
    x, y = p - BASE
    c2 = np.clip((x * x + y * y - L1**2 - L2**2) / (2 * L1 * L2), -1, 1)
    q2 = np.arccos(c2) * (1 if elbow_up else -1)
    q1 = np.arctan2(y, x) - np.arctan2(L2 * np.sin(q2), L1 + L2 * np.cos(q2))
    return np.array([q1, q2])


class Arm:
    """Joint-space PD control on a double-integrator with damping (simple dynamics)."""
    def __init__(self):
        self.q = ik(np.array([0.0, 0.35]))
        self.dq = np.zeros(2)
        self.kp, self.kd, self.dt = 120.0, 22.0, 1 / 120

    def step(self, q_ref):
        ddq = self.kp * (q_ref - self.q) - self.kd * self.dq
        self.dq += ddq * self.dt
        self.q += self.dq * self.dt


def draw_shape(img, cls, center_px, size, angle, color):
    c = np.array(center_px, float)
    if cls == "circle":
        cv2.circle(img, tuple(c.astype(int)), int(size), color, -1, cv2.LINE_AA)
        return
    n = 4 if cls == "square" else 3
    r = size * (1.35 if n == 4 else 1.5)
    pts = [c + r * np.array([np.cos(angle + 2 * np.pi * k / n), np.sin(angle + 2 * np.pi * k / n)]) for k in range(n)]
    cv2.fillPoly(img, [np.array(pts, np.int32)], color, cv2.LINE_AA)


def in_bin(px, margin=48):
    return any(np.all(np.abs(np.array(px) - w2p(b)) < margin) for b in BINS.values())


def render_scene(objects, arm=None, held=None, noise=True, rng=None, binned=()):
    rng = rng or np.random.default_rng()
    img = np.full((H, W, 3), 225, np.uint8)
    for cls, b in BINS.items():
        p = w2p(b)
        cv2.rectangle(img, tuple(p - 38), tuple(p + 38), (150, 150, 150), 3)
        cv2.putText(img, cls, (p[0] - 34, p[1] + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1)
    for o in binned:  # already-sorted objects shown (smaller) inside bins
        draw_shape(img, o["cls"], w2p(o["pos"]), o["size"] * 0.7, o["ang"], o["color"])
    for o in objects:
        if o is held:
            continue
        draw_shape(img, o["cls"], w2p(o["pos"]), o["size"], o["ang"], o["color"])
    if arm is not None:
        e, t = fk(arm.q)
        b, ep, tp = w2p(BASE), w2p(e), w2p(t)
        cv2.line(img, tuple(b), tuple(ep), (50, 50, 50), 14, cv2.LINE_AA)
        cv2.line(img, tuple(ep), tuple(tp), (90, 90, 90), 10, cv2.LINE_AA)
        cv2.circle(img, tuple(b), 14, (30, 30, 30), -1)
        cv2.circle(img, tuple(ep), 9, (30, 30, 30), -1)
        if held is not None:
            draw_shape(img, held["cls"], tp, held["size"], held["ang"], held["color"])
        cv2.circle(img, tuple(tp), 7, (0, 0, 255) if held else (0, 0, 0), 2)
    if noise:
        img = np.clip(img.astype(int) + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)
    return img


def random_objects(n, rng):
    objs, tries = [], 0
    while len(objs) < n and tries < 500:
        tries += 1
        r = rng.uniform(0.25, 0.48)
        th = rng.uniform(np.deg2rad(35), np.deg2rad(145))
        pos = np.array([r * np.cos(th), r * np.sin(th)])
        if any(np.linalg.norm(pos - o["pos"]) < 0.09 for o in objs):
            continue
        if any(np.linalg.norm(pos - b) < 0.17 for b in BINS.values()):
            continue
        cls = str(rng.choice(CLASSES))
        # color is randomized -> the AI must use SHAPE, not color
        col = tuple(int(c) for c in rng.integers(30, 200, 3))
        objs.append(dict(cls=cls, pos=pos, size=rng.uniform(13, 19), ang=rng.uniform(0, 2 * np.pi), color=col))
    return objs
