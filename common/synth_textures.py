"""Генератор СИНТЕТИЧНИХ плиток 224×224 фракцій відходів (вид згори, GSD ≈ 2 см).

Класи: mineral (бетон/цегла), wood (дошки/балки), metal (арматура, листи), mixed, background.
Кожна «сцена» має власний колірний зсув, освітлення, розмиття й шум — так моделюється
доменний зсув між знімками, а групове розбиття за сценами має сенс.
"""
import numpy as np, cv2

CLASSES = ["mineral", "wood", "metal", "mixed", "background"]
S = 224


def _poly(rng, cx, cy, r, n=None, irregular=.45):
    n = n or rng.integers(4, 8)
    a = np.sort(rng.uniform(0, 2 * np.pi, n))
    rr = r * rng.uniform(1 - irregular, 1, n)
    return np.stack([cx + rr * np.cos(a), cy + rr * np.sin(a)], 1).astype(np.int32)


def _base(rng, kind):
    if kind == "dust":
        col = rng.uniform([135, 130, 125], [175, 170, 160])
    else:  # ґрунт / трава
        col = rng.choice([[120, 100, 75], [95, 120, 70], [140, 125, 100], [85, 85, 80]]) + rng.normal(0, 8, 3)
    img = np.ones((S, S, 3), np.float32) * col
    low = cv2.resize(rng.normal(0, 1, (12, 12)).astype(np.float32), (S, S), interpolation=cv2.INTER_CUBIC)
    return img + low[..., None] * 14 + rng.normal(0, 9, (S, S, 3))


def _mineral(img, rng, n):
    for _ in range(n):
        brick = rng.random() < .35
        col = (rng.uniform([150, 70, 50], [185, 100, 75]) if brick else rng.uniform([120, 120, 115], [185, 182, 175]))
        p = _poly(rng, rng.uniform(0, S), rng.uniform(0, S), rng.uniform(8, 26))
        cv2.fillPoly(img, [p + 2], (col * .55).tolist()); cv2.fillPoly(img, [p], col.tolist())


def _wood(img, rng, n):
    for _ in range(n):
        L, W = rng.uniform(60, 200), rng.uniform(8, 22); ang = rng.uniform(0, np.pi)
        cx, cy = rng.uniform(0, S), rng.uniform(0, S)
        col = rng.uniform([120, 80, 45], [185, 140, 90])
        box = cv2.boxPoints(((cx, cy), (L, W), np.degrees(ang))).astype(np.int32)
        cv2.fillPoly(img, [box + 2], (col * .5).tolist()); cv2.fillPoly(img, [box], col.tolist())
        for k in range(3):  # волокна
            off = (k - 1) * W / 4
            dx, dy = np.cos(ang) * L / 2, np.sin(ang) * L / 2; ox, oy = -np.sin(ang) * off, np.cos(ang) * off
            cv2.line(img, (int(cx - dx + ox), int(cy - dy + oy)), (int(cx + dx + ox), int(cy + dy + oy)), (col * .8).tolist(), 1)


def _metal(img, rng, n):
    for _ in range(n):
        if rng.random() < .6:   # арматура
            x0, y0 = rng.uniform(0, S, 2); ang = rng.uniform(0, np.pi); L = rng.uniform(60, 220)
            col = rng.choice([[70, 60, 55], [120, 70, 40], [95, 90, 88]]) + rng.normal(0, 8, 3)
            p1 = (int(x0), int(y0)); p2 = (int(x0 + L * np.cos(ang)), int(y0 + L * np.sin(ang)))
            cv2.line(img, p1, p2, (col * .5).tolist(), 4); cv2.line(img, p1, p2, col.tolist(), 2)
        else:                   # лист / профіль з відблиском
            box = cv2.boxPoints(((rng.uniform(0, S), rng.uniform(0, S)), (rng.uniform(30, 90), rng.uniform(20, 60)),
                                 rng.uniform(0, 180))).astype(np.int32)
            col = rng.uniform([140, 145, 150], [200, 205, 210])
            cv2.fillPoly(img, [box], col.tolist())
            c = box.mean(0).astype(int); cv2.circle(img, tuple(int(v) for v in c), int(rng.uniform(4, 10)), (240, 240, 240), -1)
            if rng.random() < .5:  # іржа
                cv2.circle(img, tuple(int(v) for v in box[0]), int(rng.uniform(4, 12)), (130, 70, 35), -1)


def make_tile(cls, rng, style):
    if cls == "background":
        img = _base(rng, "soil")
        if rng.random() < .5: _mineral(img, rng, rng.integers(0, 3))        # поодиноке каміння
    elif cls == "mineral":
        img = _base(rng, "dust"); _mineral(img, rng, rng.integers(25, 50))
        if rng.random() < .3: _wood(img, rng, 1)
        if rng.random() < .3: _metal(img, rng, 1)
    elif cls == "wood":
        img = _base(rng, rng.choice(["dust", "soil"])); _wood(img, rng, rng.integers(5, 12))
        if rng.random() < .4: _mineral(img, rng, rng.integers(3, 10))
    elif cls == "metal":
        img = _base(rng, "dust"); _metal(img, rng, rng.integers(8, 18))
        if rng.random() < .5: _mineral(img, rng, rng.integers(5, 15))
    else:  # mixed
        img = _base(rng, "dust")
        _mineral(img, rng, rng.integers(10, 25)); _wood(img, rng, rng.integers(2, 5)); _metal(img, rng, rng.integers(3, 8))
    if rng.random() < .35:   # тінь
        x = int(rng.uniform(0, S)); img[:, x:] *= rng.uniform(.55, .8)
    img = img * style["gain"] + style["bias"]
    img = cv2.GaussianBlur(img, (0, 0), style["blur"]) + rng.normal(0, style["noise"], img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def scene_style(rng):
    return {"gain": rng.uniform(.8, 1.15, 3), "bias": rng.uniform(-15, 15, 3),
            "blur": rng.uniform(.5, 1.6), "noise": rng.uniform(3, 10)}
