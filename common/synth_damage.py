"""Генератор СИНТЕТИЧНИХ знімків «після руйнувань» (вид згори) у форматі, подібному до xBD.

Кожне зображення: вулиці, трава, дерева, тіні та будівлі трьох станів — ціла (no-damage),
з провалом даху (major-damage → roof_collapse) і зруйнована, з завалом (destroyed → rubble).
Розмітка зберігається як JSON з полігонами контурів будівель (WKT) і полем subtype, як у xBD,
тому подальший код ноутбука однаковий для синтетичних і справжніх даних xBD.
"""
import json, os
import numpy as np, cv2


def _rot_rect(cx, cy, w, h, ang):
    return cv2.boxPoints(((cx, cy), (w, h), ang)).astype(np.float32)


def _blob(rng, cx, cy, r, n=16):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rr = r * rng.uniform(.7, 1.15, n)
    return np.stack([cx + rr * np.cos(a), cy + rr * np.sin(a)], 1).astype(np.int32)


def make_image(rng, style, size=384):
    gsd = style["gsd"]; g = 1 / gsd
    img = np.ones((size, size, 3), np.float32) * style["ground"]
    img += cv2.resize(rng.normal(0, 1, (10, 10)).astype(np.float32), (size, size), interpolation=cv2.INTER_CUBIC)[..., None] * 12
    # вулиці
    for _ in range(rng.integers(1, 3)):
        if rng.random() < .5:
            y = int(rng.uniform(0, size)); cv2.line(img, (0, y), (size, y), style["road"], int(8 * g))
        else:
            x = int(rng.uniform(0, size)); cv2.line(img, (x, 0), (x, size), style["road"], int(8 * g))
    feats = []
    occupied = np.zeros((size, size), np.uint8)
    ang0 = style["angle"]
    for _ in range(rng.integers(14, 26)):
        large = rng.random() < .15
        w, h = (rng.uniform(30, 60, 2) if large else rng.uniform(9, 22, 2)) * g
        cx, cy = rng.uniform(w / 2, size - w / 2), rng.uniform(h / 2, size - h / 2)
        ang = ang0 + rng.normal(0, 4)
        box = _rot_rect(cx, cy, w, h, ang)
        m = np.zeros_like(occupied); cv2.fillPoly(m, [box.astype(np.int32)], 1)
        if (m & occupied).sum() > 0: continue
        cv2.dilate(m, np.ones((5, 5), np.uint8), dst=m); occupied |= m
        state = rng.choice(["no-damage", "major-damage", "destroyed"], p=style["p"])
        roof = np.array(style["roofs"][rng.integers(len(style["roofs"]))], np.float32) + rng.normal(0, 8, 3)
        sh = np.array(style["sun"]) * g * rng.uniform(1.5, 3)
        if state != "destroyed":
            shimg = img.copy(); cv2.fillPoly(shimg, [(box + sh).astype(np.int32)], (30, 30, 30))
            img = cv2.addWeighted(img, .5, shimg, .5, 0)
            cv2.fillPoly(img, [box.astype(np.int32)], roof.tolist())
            cv2.line(img, tuple(((box[0] + box[1]) / 2).astype(int)), tuple(((box[2] + box[3]) / 2).astype(int)),
                     (roof * .8).tolist(), 1)                                      # конек даху
            if state == "major-damage":                                          # провал даху
                for _ in range(rng.integers(1, 3)):
                    px, py = cx + rng.uniform(-.25, .25) * w, cy + rng.uniform(-.25, .25) * h
                    hole = _blob(rng, px, py, rng.uniform(.2, .35) * min(w, h))
                    cv2.fillPoly(img, [hole], rng.uniform(25, 60, 3).tolist())
                    for _ in range(rng.integers(3, 8)):
                        q = _blob(rng, px + rng.normal(0, 3), py + rng.normal(0, 3), rng.uniform(1, 2.5), 6)
                        cv2.fillPoly(img, [q], rng.uniform(100, 180, 3).tolist())
        else:                                                                    # завал
            k = rng.uniform(1.0, 1.3)
            cv2.fillPoly(img, [_blob(rng, cx, cy, k * max(w, h) / 2)], (rng.uniform(95, 135, 3)).tolist())
            for _ in range(int(w * h / 25)):
                q = _blob(rng, cx + rng.normal(0, w / 3), cy + rng.normal(0, h / 3), rng.uniform(1, 3.5), 6)
                col = rng.choice([roof, [150, 145, 140], [90, 80, 70], [180, 170, 160]]) * rng.uniform(.8, 1.1)
                cv2.fillPoly(img, [q], np.asarray(col).tolist())
        feats.append({"properties": {"feature_type": "building", "subtype": state},
                      "wkt": "POLYGON ((" + ", ".join(f"{x:.2f} {y:.2f}" for x, y in np.vstack([box, box[:1]])) + "))"})
    for _ in range(rng.integers(5, 20)):                                         # дерева
        cx, cy, r = rng.uniform(0, size), rng.uniform(0, size), rng.uniform(1.5, 4) * g
        if occupied[int(min(cy, size - 1)), int(min(cx, size - 1))]: continue
        cv2.circle(img, (int(cx + 2), int(cy + 2)), int(r), (40, 45, 35), -1)
        cv2.circle(img, (int(cx), int(cy)), int(r), (rng.uniform(40, 70), rng.uniform(80, 120), rng.uniform(35, 60)), -1)
    img = img * style["light"] + rng.normal(0, style["noise"], img.shape)
    img = cv2.GaussianBlur(img, (0, 0), style["blur"])
    return np.clip(img, 0, 255).astype(np.uint8), feats


def make_dataset(root, n_events=6, per_event=40, size=384, seed=0):
    rng = np.random.default_rng(seed)
    os.makedirs(f"{root}/images", exist_ok=True); os.makedirs(f"{root}/labels", exist_ok=True)
    names = ["quake-a", "quake-b", "tornado-a", "flood-a", "shelling-a", "shelling-b"]
    for e in range(n_events):
        style = {"gsd": float(rng.uniform(.45, .6)), "ground": rng.uniform([90, 100, 70], [150, 145, 120]),
                 "road": tuple(float(v) for v in rng.uniform(70, 110, 3)), "angle": float(rng.uniform(0, 90)),
                 "roofs": [rng.uniform([150, 60, 50], [200, 100, 80]), rng.uniform([110, 110, 110], [170, 170, 175]),
                           rng.uniform([90, 70, 60], [140, 110, 90])],
                 "sun": list(rng.normal(0, 1, 2) / 1.4), "light": float(rng.uniform(.85, 1.15)),
                 "noise": float(rng.uniform(3, 8)), "blur": float(rng.uniform(.5, 1.0)),
                 "p": [.55, .25, .20]}
        for i in range(per_event):
            img, feats = make_image(rng, style, size)
            name = f"{names[e]}_{i:08d}_post_disaster"
            cv2.imwrite(f"{root}/images/{name}.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            json.dump({"metadata": {"disaster": names[e], "gsd": style["gsd"], "img_name": name + ".png"},
                       "features": {"xy": feats}}, open(f"{root}/labels/{name}.json", "w"))
