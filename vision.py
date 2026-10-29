"""Vision AI: contour detection + learned shape classifier (MLP, scikit-learn)."""
import numpy as np
import cv2
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sim import CLASSES, draw_shape, p2w, in_bin

PATCH = 32


def to_feat(patch_gray):
    # binarize -> pure silhouette (color/brightness invariant)
    b = cv2.GaussianBlur(patch_gray, (3, 3), 0)
    _, b = cv2.threshold(b, 205, 255, cv2.THRESH_BINARY_INV)
    p = cv2.resize(b, (PATCH, PATCH), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    return p.ravel()


def crop_feat(img, c, half=26):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x, y = int(c[0]), int(c[1])
    pad = cv2.copyMakeBorder(g, half, half, half, half, cv2.BORDER_CONSTANT, value=225)
    return to_feat(pad[y:y + 2 * half, x:x + 2 * half])


def make_dataset(n_per_class=1500, seed=0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for ci, cls in enumerate(CLASSES):
        for _ in range(n_per_class):
            img = np.full((64, 64, 3), 225, np.uint8)
            col = tuple(int(c) for c in rng.integers(30, 200, 3))
            c = (32 + rng.uniform(-3, 3), 32 + rng.uniform(-3, 3))
            draw_shape(img, cls, c, rng.uniform(12, 20), rng.uniform(0, 2 * np.pi), col)
            img = np.clip(img.astype(int) + rng.normal(0, 8, img.shape), 0, 255).astype(np.uint8)
            X.append(crop_feat(img, (32, 32)))
            y.append(ci)
    return np.array(X), np.array(y)


def train(path="data/shape_mlp.joblib"):
    X, y = make_dataset()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200, random_state=0)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    rep = classification_report(yte, pred, target_names=CLASSES, digits=3)
    cm = confusion_matrix(yte, pred)
    joblib.dump(clf, path)
    return clf, rep, cm


def detect(img, clf):
    """Return list of dict(pos_world, px, cls, conf)."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (5, 5), 0)
    _, m = cv2.threshold(g, 205, 255, cv2.THRESH_BINARY_INV)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < 250 or a > 3000:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w > 80 or h > 80:
            continue
        M = cv2.moments(c)
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        if in_bin((cx, cy)):  # ignore objects already sorted
            continue
        p = clf.predict_proba([crop_feat(img, (cx, cy))])[0]
        out.append(dict(px=(cx, cy), pos=p2w((cx, cy)), cls=CLASSES[int(p.argmax())], conf=float(p.max())))
    return out


if __name__ == "__main__":
    _, rep, cm = train()
    print(rep)
    print(cm)
