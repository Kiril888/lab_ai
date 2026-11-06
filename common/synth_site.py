"""Генератор СИНТЕТИЧНИХ ортофотопланів будмайданчика (вид згори) з технікою.

Використовується в ЛР1–ЛР2 замість реальних знімків: кожна сцена — ґрунтовий фон, дороги,
купи відходів, тіні та машини 4 класів з еталонними рамками (bbox у пікселях).
Розміри машин задано в метрах і переведено в пікселі через GSD.
"""
import numpy as np, cv2

CLASSES = ["dump_truck", "excavator", "crane", "other_vehicle"]


def _rect(cx, cy, L, W, ang):
    """Кути прямокутника довжини L і ширини W з центром (cx, cy), поворот ang (рад)."""
    c, s = np.cos(ang), np.sin(ang)
    pts = np.array([[-L / 2, -W / 2], [L / 2, -W / 2], [L / 2, W / 2], [-L / 2, W / 2]])
    R = np.array([[c, -s], [s, c]])
    return pts @ R.T + [cx, cy]


def _part(cx, cy, ang, dx, dy, L, W):
    """Частина машини, зміщена на (dx, dy) у локальних координатах машини."""
    c, s = np.cos(ang), np.sin(ang)
    return _rect(cx + dx * c - dy * s, cy + dx * s + dy * c, L, W, ang)


def _fill(img, poly, color):
    cv2.fillPoly(img, [np.round(poly).astype(np.int32)], color, lineType=cv2.LINE_AA)


def _machine(rng, cls, cx, cy, ang, g):
    """Повертає список (полігон, колір) у пікселях; g = пікселів на метр."""
    yellow = tuple(int(v) for v in rng.integers([200, 150, 0], [255, 200, 50]))
    parts = []
    if cls == "dump_truck":
        L, W = rng.uniform(8, 10), rng.uniform(2.4, 2.6)
        cab = tuple(int(v) for v in rng.choice([[230, 230, 230], [200, 40, 30], [240, 180, 20], [40, 80, 160]]))
        body = tuple(int(v) for v in rng.choice([[120, 120, 120], [190, 110, 40], [90, 90, 95]]))
        parts += [(_part(cx, cy, ang, -L / 2 + 1.1, 0, 2.2, W, ) * g, cab)]
        parts += [(_part(cx, cy, ang, 1.1, 0, L - 2.2, W) * g, body)]
        if rng.random() < .6:  # вантаж у кузові
            load = tuple(int(v) for v in rng.choice([[150, 130, 110], [110, 100, 90], [170, 90, 60]]))
            parts += [(_part(cx, cy, ang, 1.1, 0, L - 3.0, W - .6) * g, load)]
    elif cls == "excavator":
        W = rng.uniform(2.8, 3.2); arm = rng.uniform(5, 8); arm_ang = ang + rng.uniform(-1.2, 1.2)
        parts += [(_part(cx, cy, ang, 0, -W / 2 + .3, 4.5, .6) * g, (40, 40, 40)),   # гусениці
                  (_part(cx, cy, ang, 0, W / 2 - .3, 4.5, .6) * g, (40, 40, 40)),
                  (_part(cx, cy, ang, 0, 0, 3.4, W - .4) * g, yellow)]
        ax, ay = cx + 1.2 * np.cos(arm_ang), cy + 1.2 * np.sin(arm_ang)
        parts += [(_part(ax, ay, arm_ang, arm / 2, 0, arm, .7) * g, yellow),
                  (_part(ax, ay, arm_ang, arm + .5, 0, 1.2, 1.4) * g, (60, 60, 60))]  # ківш
    elif cls == "crane":
        L, W = rng.uniform(11, 13), rng.uniform(2.5, 2.7); boom = rng.uniform(8, 18)
        carrier = tuple(int(v) for v in rng.choice([[230, 230, 230], [200, 40, 30], [240, 180, 20]]))
        parts += [(_part(cx, cy, ang, 0, 0, L, W) * g, carrier),
                  (_part(cx, cy, ang, -L / 2 + 1, 0, 2, W) * g, (90, 90, 90)),
                  (_part(cx, cy, ang, boom / 2 - 1, 0, boom, .8) * g, yellow)]
        for s in (-1, 1):  # аутригери
            parts += [(_part(cx, cy, ang, -L / 4, s * (W / 2 + .6), .5, 1.4) * g, (70, 70, 70)),
                      (_part(cx, cy, ang, L / 4, s * (W / 2 + .6), .5, 1.4) * g, (70, 70, 70))]
    else:  # other_vehicle: легковик, навантажувач або бетонозмішувач
        kind = rng.choice(["car", "loader", "mixer"], p=[.5, .3, .2])
        if kind == "car":
            col = tuple(int(v) for v in rng.integers(20, 240, 3))
            parts += [(_part(cx, cy, ang, 0, 0, 4.5, 1.8) * g, col),
                      (_part(cx, cy, ang, -.3, 0, 2.0, 1.5) * g, tuple(int(v * .6) for v in col))]
        elif kind == "loader":
            parts += [(_part(cx, cy, ang, 0, 0, 6, 2.4) * g, yellow),
                      (_part(cx, cy, ang, 3.4, 0, .9, 2.8) * g, (60, 60, 60))]
        else:
            parts += [(_part(cx, cy, ang, -3, 0, 2.2, 2.5) * g, (230, 230, 230)),
                      (_part(cx, cy, ang, 1.3, 0, 5.5, 2.3) * g, (180, 180, 185)),
                      (_part(cx, cy, ang, 1.3, 0, 4.5, 1.2) * g, (150, 150, 155))]
    return parts


def make_site(H=1920, W=1920, gsd=0.05, seed=0, n_machines=None, style=None):
    """Синтетична сцена: повертає (img uint8 H×W×3, objects=[(cls_id, x0, y0, x1, y1), ...])."""
    rng = np.random.default_rng(seed)
    g = 1.0 / gsd
    style = style or {}
    base = np.array(style.get("soil", rng.uniform([120, 100, 75], [170, 150, 120])))
    light = style.get("light", rng.uniform(.8, 1.15))
    img = np.ones((H, W, 3), np.float32) * base
    low = cv2.resize(rng.normal(0, 1, (H // 32 + 1, W // 32 + 1)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    img += low[..., None] * 15 + rng.normal(0, 7, (H, W, 3))
    img = np.clip(img, 0, 255).astype(np.uint8)
    # дороги
    for _ in range(rng.integers(1, 3)):
        y = int(rng.uniform(.1, .9) * H); th = int(rng.uniform(5, 8) * g)
        ang = rng.uniform(-.4, .4)
        cv2.line(img, (0, y), (W, int(y + W * np.tan(ang))), tuple(int(v) for v in rng.integers(85, 115, 3)), th)
    # купи відходів (фонові об'єкти — схожі текстури, але не техніка)
    for _ in range(rng.integers(4, 10)):
        cx, cy = rng.uniform(0, W), rng.uniform(0, H); r = rng.uniform(3, 9) * g
        n = 14; a = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pts = np.stack([cx + r * rng.uniform(.6, 1.1, n) * np.cos(a), cy + r * rng.uniform(.6, 1.1, n) * np.sin(a)], 1)
        col = tuple(int(v) for v in rng.choice([[130, 125, 120], [150, 100, 70], [100, 95, 85], [175, 170, 160]]))
        _fill(img, pts, col)
        m = np.zeros((H, W), np.uint8); _fill(m, pts, 255)
        speck = rng.normal(0, 25, (H, W, 1))
        img = np.where(m[..., None] > 0, np.clip(img + speck, 0, 255), img).astype(np.uint8)
    # техніка
    objects = []
    n = n_machines if n_machines is not None else rng.integers(6, 14)
    p = style.get("class_p", [.4, .25, .1, .25])
    sun = rng.uniform(0, 2 * np.pi); sh = np.array([np.cos(sun), np.sin(sun)]) * rng.uniform(.8, 1.6) * g
    for _ in range(n):
        cls = rng.choice(4, p=p)
        cx, cy = rng.uniform(.05, .95) * W / g, rng.uniform(.05, .95) * H / g
        ang = rng.choice([0, np.pi / 2]) if style.get("axis") else rng.uniform(0, 2 * np.pi)
        parts = _machine(rng, CLASSES[cls], cx, cy, ang, g)
        for poly, _ in parts:      # тінь
            sh_img = img.copy(); _fill(sh_img, poly + sh, (0, 0, 0)); img = cv2.addWeighted(img, .55, sh_img, .45, 0)
        for poly, col in parts:
            _fill(img, poly, col)
        allp = np.concatenate([q for q, _ in parts])
        x0, y0 = np.clip(allp.min(0), 0, [W, H]); x1, y1 = np.clip(allp.max(0), 0, [W, H])
        if (x1 - x0) > 4 and (y1 - y0) > 4:
            objects.append((int(cls), float(x0), float(y0), float(x1), float(y1)))
    img = np.clip(img.astype(np.float32) * light + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (0, 0), style.get("blur", .7))
    return img, objects


def clip_to_tile(objects, x0, y0, size, min_visible=0.5):
    """Переносить рамки сцени в плитку; лишає об'єкти, видимі щонайменше на min_visible площі."""
    out = []
    for c, a0, b0, a1, b1 in objects:
        ix0, iy0, ix1, iy1 = max(a0, x0), max(b0, y0), min(a1, x0 + size), min(b1, y0 + size)
        if ix1 <= ix0 or iy1 <= iy0: continue
        if (ix1 - ix0) * (iy1 - iy0) / ((a1 - a0) * (b1 - b0)) < min_visible: continue
        if min(ix1 - ix0, iy1 - iy0) < 10: continue        # правило інструкції: менша сторона ≥ 10 пкс
        out.append((c, ix0 - x0, iy0 - y0, ix1 - x0, iy1 - y0))
    return out
