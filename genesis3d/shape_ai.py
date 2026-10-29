"""Shape AI for the Genesis 3D version (no Genesis dependency -> testable anywhere).

Input : binary silhouette mask of ONE object (from the depth camera: "height above table")
Output: class in {circle, square, triangle} + confidence
Feature is scale/translation normalized (tight square crop -> 32x32), so the
same classifier works for any camera distance. Trained on synthetic silhouettes
with perspective-like distortion (anisotropic scale + shear) to cover the
slight side-face visibility of a real top-down camera.
"""
import numpy as np
import cv2
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

CLASSES = ["circle", "square", "triangle"]
PATCH = 32


def mask_feature(mask):
    """mask: uint8 {0,255} image containing one blob -> 1024-dim feature."""
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    side = int(max(x1 - x0, y1 - y0) * 1.2) + 2
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    pad = cv2.copyMakeBorder(mask, side, side, side, side, cv2.BORDER_CONSTANT, value=0)
    crop = pad[cy + side - side // 2: cy + side + side // 2, cx + side - side // 2: cx + side + side // 2]
    return cv2.resize(crop, (PATCH, PATCH), interpolation=cv2.INTER_AREA).astype(np.float32).ravel() / 255.0


def synth_mask(cls, rng, size=128):
    img = np.zeros((size, size), np.uint8)
    c = np.array([size / 2, size / 2])
    r = rng.uniform(18, 40)
    ang = rng.uniform(0, 2 * np.pi)
    if cls == "circle":
        pts = [c + r * np.array([np.cos(t), np.sin(t)]) for t in np.linspace(0, 2 * np.pi, 64, endpoint=False)]
    else:
        n = 4 if cls == "square" else 3
        pts = [c + r * np.array([np.cos(ang + 2 * np.pi * k / n), np.sin(ang + 2 * np.pi * k / n)]) for k in range(n)]
    pts = np.array(pts) - c
    # perspective-like distortion: anisotropic scale + shear
    A = np.array([[rng.uniform(0.88, 1.12), rng.uniform(-0.12, 0.12)],
                  [rng.uniform(-0.12, 0.12), rng.uniform(0.88, 1.12)]])
    pts = pts @ A.T + c
    cv2.fillPoly(img, [pts.astype(np.int32)], 255, cv2.LINE_AA)
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    # edge noise (depth sensor-like)
    if rng.random() < 0.7:
        k = np.ones((3, 3), np.uint8)
        img = cv2.erode(img, k) if rng.random() < 0.5 else cv2.dilate(img, k)
    return img


def make_dataset(n_per_class=2000, seed=0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for ci, cls in enumerate(CLASSES):
        for _ in range(n_per_class):
            X.append(mask_feature(synth_mask(cls, rng)))
            y.append(ci)
    return np.array(X), np.array(y)


def train(path="shape_mlp_3d.joblib"):
    X, y = make_dataset()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200, random_state=0).fit(Xtr, ytr)
    rep = classification_report(yte, clf.predict(Xte), target_names=CLASSES, digits=3)
    joblib.dump(clf, path)
    return clf, rep


def classify(clf, mask):
    p = clf.predict_proba([mask_feature(mask)])[0]
    return CLASSES[int(p.argmax())], float(p.max())


if __name__ == "__main__":
    _, rep = train()
    print(rep)
