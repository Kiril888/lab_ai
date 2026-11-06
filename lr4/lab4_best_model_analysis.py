display = print
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as _plt
_orig = _plt.savefig
def _sv(name, *a, **k): return _orig("best_" + name, *a, **k)
_plt.savefig = _sv
# %% [markdown]
# # ЛР4 — детектування провалів дахів і завалів на знімках після руйнувань
# Варіант 9: **RetinaNet** (torchvision, ResNet-50 FPN v2), дослідження: **зменшення у 2 рази**.
#
# **Дані:** у цій версії набір **синтетичний**, у форматі xBD: 6 «подій» × 40 знімків «після»
# 384×384 (GSD 0,45–0,6 м), згенерованих скриптом `synth_damage.py`. Розмітка — полігони контурів
# будівель зі станом `no-damage / major-damage / destroyed`, як у xBD. Класи детектора:
# `major-damage` → **roof_collapse** (провал даху), `destroyed` → **rubble** (завал).
# Для справжнього xBD (xview2.org) досить розпакувати архів у теку `xbd/`.
#
# Ваги: екстрактор ResNet-50, переднавчений на ImageNet (файл `resnet50_ram-a26f946b.pth`);
# за його відсутності — torchvision COCO `retinanet_resnet50_fpn_v2(weights="DEFAULT")`.

# %%
pass
import os, glob, json, random, re
from functools import partial
import numpy as np, pandas as pd, matplotlib.pyplot as plt, cv2
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection import retinanet_resnet50_fpn_v2
from torchvision.models.detection.retinanet import RetinaNetClassificationHead

SMOKE = os.environ.get("SMOKE") == "1"
PROTOCOL = dict(seed=0, img_size=384, epochs=15, epochs_baseline=5, batch=4,
                optimizer="SGD", lr=0.005, momentum=0.9, weight_decay=1e-4, patience=5,
                aug="hflip, vflip, rot90", score_thresh_eval=0.05, nms_iou=0.5)
CLASSES = ["roof_collapse", "rubble"]           # id у моделі: 1, 2 (0 — фон)
MAP = {"major-damage": 1, "destroyed": 2}
DEV = "cuda" if torch.cuda.is_available() else "cpu"
def set_seed(s): random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
set_seed(PROTOCOL["seed"]); json.dump(PROTOCOL, open("protocol.json", "w"), indent=1); print(DEV)

# %% [markdown]
# ## 1. Конвертація розмітки xBD → рамки (COCO-подібний список)

# %%
XBD = "xbd"          # тека з images/ і labels/ (xBD або синтетичні)
import sys; sys.path.append("../common"); sys.path.append(".")
if not os.path.isdir(f"{XBD}/labels"):
    from synth_damage import make_dataset
    make_dataset(XBD, n_events=6, per_event=40, size=PROTOCOL["img_size"], seed=0)
rows = []
for jf in sorted(glob.glob(f"{XBD}/labels/*_post_disaster.json")):
    j = json.load(open(jf)); name = os.path.basename(jf).replace(".json", ".png")
    event = j["metadata"].get("disaster", name.split("_")[0])
    gsd = j["metadata"].get("gsd", np.nan)
    boxes, labels = [], []
    for f in j["features"]["xy"]:
        c = MAP.get(f["properties"].get("subtype"))
        if c is None: continue
        pts = np.array(re.findall(r"(-?[\d.]+) (-?[\d.]+)", f["wkt"]), float)
        x0, y0 = pts.min(0); x1, y1 = pts.max(0)
        if min(x1 - x0, y1 - y0) >= 4: boxes.append([x0, y0, x1, y1]); labels.append(c)
    rows.append(dict(file=f"{XBD}/images/{name}", event=event, gsd=gsd,
                     boxes=json.dumps(boxes), labels=json.dumps(labels), n=len(boxes)))
ann = pd.DataFrame(rows)
# лишаємо зображення з об'єктами + ~20 % без них (негативні приклади)
pos = ann[ann.n > 0]; neg = ann[ann.n == 0].sample(min(len(ann[ann.n == 0]), len(pos) // 4), random_state=0)
ann = pd.concat([pos, neg]).reset_index(drop=True)
print(len(ann), "зображень;", ann.event.nunique(), "подій"); display(ann.groupby("event").n.agg(["size", "sum"]))

# %% [markdown]
# ### Групове розбиття за подіями (витік між подіями неможливий) і статистика

# %%
from sklearn.model_selection import GroupShuffleSplit
i_tr, i_rest = next(GroupShuffleSplit(1, test_size=0.3, random_state=0).split(ann, groups=ann.event))
rest = ann.iloc[i_rest]
i_va, i_te = next(GroupShuffleSplit(1, test_size=0.5, random_state=0).split(rest, groups=rest.event))
ann["split"] = "train"; ann.loc[rest.index[i_va], "split"] = "val"; ann.loc[rest.index[i_te], "split"] = "test"
ann.to_csv("det_split.csv", index=False)
print(ann.groupby("split").event.unique())

def size_group(b):
    a = (b[2] - b[0]) * (b[3] - b[1]); return "S" if a < 32 ** 2 else ("M" if a < 96 ** 2 else "L")
st = []
for r in ann.itertuples():
    for b, l in zip(json.loads(r.boxes), json.loads(r.labels)):
        st.append(dict(split=r.split, cls=CLASSES[l - 1], w=b[2] - b[0], h=b[3] - b[1], size=size_group(b)))
st = pd.DataFrame(st)
display(pd.crosstab(st.cls, st.split)); display(pd.crosstab(st["size"], st.split, normalize="columns").round(3))
st[["w", "h"]].plot.hist(bins=40, alpha=.6, title="Сторони рамок, пкс"); plt.savefig("fig_box_sizes.png", dpi=150); plt.show()

# %%
def draw(img, boxes, labels, color=(0, 255, 0), scores=None):
    img = img.copy()
    for k, (b, l) in enumerate(zip(boxes, labels)):
        x0, y0, x1, y1 = map(int, b); cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
        t = CLASSES[l - 1] + (f" {scores[k]:.2f}" if scores is not None else "")
        cv2.putText(img, t, (x0, max(10, y0 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return img

sample = ann[ann.n > 0].sample(min(12, (ann.n > 0).sum()), random_state=1)
fig, ax = plt.subplots(3, 4, figsize=(16, 12))
for a, r in zip(ax.ravel(), sample.itertuples()):
    im = cv2.cvtColor(cv2.imread(r.file), cv2.COLOR_BGR2RGB)
    a.imshow(draw(im, json.loads(r.boxes), json.loads(r.labels))); a.set_title(r.event, fontsize=8); a.axis("off")
plt.tight_layout(); plt.savefig("fig_labels_check.png", dpi=120); plt.show()

# %% [markdown]
# ## 2. Набір даних з аугментаціями (вид згори: віддзеркалення + поворот на 90°)

# %%
class DetDS(Dataset):
    def __init__(self, frame, train=False, factor=1):
        self.r = frame.reset_index(drop=True); self.train = train; self.factor = factor
    def __len__(self): return len(self.r)
    def __getitem__(self, i):
        r = self.r.iloc[i]
        img = cv2.cvtColor(cv2.imread(r.file), cv2.COLOR_BGR2RGB)
        if self.factor > 1: img = emulate_gsd(img, self.factor)
        b = np.array(json.loads(r.boxes), np.float32).reshape(-1, 4); l = np.array(json.loads(r.labels), np.int64)
        H, W = img.shape[:2]
        if self.train:
            if random.random() < .5: img = img[:, ::-1]; b[:, [0, 2]] = W - b[:, [2, 0]]
            if random.random() < .5: img = img[::-1]; b[:, [1, 3]] = H - b[:, [3, 1]]
            if random.random() < .5:   # поворот на 90° проти год. стрілки (квадратні знімки)
                img = np.rot90(img); b = np.stack([b[:, 1], W - b[:, 2], b[:, 3], W - b[:, 0]], 1)
        x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255
        return x, {"boxes": torch.from_numpy(b).float(), "labels": torch.from_numpy(l)}

def emulate_gsd(img, factor):          # лістинг 4.3
    h, w = img.shape[:2]
    small = cv2.resize(img, (w // factor, h // factor), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

collate = lambda b: tuple(zip(*b))
def dl(split, train=False, factor=1, bs=None):
    return DataLoader(DetDS(ann[ann.split == split], train, factor), batch_size=bs or PROTOCOL["batch"],
                      shuffle=train, collate_fn=collate, num_workers=2)

# %% [markdown]
# ## 3. Модель RetinaNet і навчання (baseline = лише голова; основна = донавчання всієї мережі)

# %%
from torchvision.models import resnet50
from torchvision.ops import FrozenBatchNorm2d
from torchvision.ops.feature_pyramid_network import LastLevelP6P7
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.models.detection.retinanet import RetinaNet, RetinaNetHead, _default_anchorgen
BACKBONE_W = "resnet50_ram-a26f946b.pth"
def make_model(freeze_backbone=False):
    kw = dict(min_size=PROTOCOL["img_size"], max_size=PROTOCOL["img_size"], score_thresh=PROTOCOL["score_thresh_eval"],
              nms_thresh=PROTOCOL["nms_iou"], detections_per_img=300)
    if os.path.exists(BACKBONE_W):          # ResNet-50 (ImageNet) + FPN P3–P7 + голови RetinaNet v2
        r = resnet50(norm_layer=FrozenBatchNorm2d)
        r.load_state_dict(torch.load(BACKBONE_W, map_location="cpu"), strict=False)
        bb = _resnet_fpn_extractor(r, 3, returned_layers=[2, 3, 4], extra_blocks=LastLevelP6P7(2048, 256))
        ag = _default_anchorgen()
        head = RetinaNetHead(256, ag.num_anchors_per_location()[0], len(CLASSES) + 1, norm_layer=partial(nn.GroupNorm, 32))
        head.regression_head._loss_type = "giou"
        m = RetinaNet(bb, len(CLASSES) + 1, anchor_generator=ag, head=head, **kw)
    else:                                   # COCO-переднавчений RetinaNet v2 з torchvision
        m = retinanet_resnet50_fpn_v2(weights="DEFAULT", **kw)
        na = m.head.classification_head.num_anchors
        m.head.classification_head = RetinaNetClassificationHead(256, na, len(CLASSES) + 1,
                                                                 norm_layer=partial(nn.GroupNorm, 32))
    if freeze_backbone:
        for p in m.backbone.parameters(): p.requires_grad = False
    return m.to(DEV)

@torch.no_grad()
def predict(m, loader):
    m.eval(); P, G = [], []
    for xs, ts in loader:
        out = m([x.to(DEV) for x in xs])
        P += [{k: v.cpu() for k, v in o.items()} for o in out]; G += list(ts)
    return P, G

from torchmetrics.detection import MeanAveragePrecision
def coco_metrics(P, G):
    mp = MeanAveragePrecision(iou_type="bbox", class_metrics=True); mp.update(P, G)
    r = mp.compute(); return {k: (v.tolist() if v.numel() > 1 else float(v)) for k, v in r.items()}

def train(name, m, epochs):
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=PROTOCOL["lr"], momentum=PROTOCOL["momentum"], weight_decay=PROTOCOL["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    tr, va = dl("train", True), dl("val")
    hist, best, bad = [], -1, 0
    for ep in range(epochs):
        m.train(); tot = {"classification": 0., "bbox_regression": 0.}; n = 0
        for xs, ts in tr:
            losses = m([x.to(DEV) for x in xs], [{k: v.to(DEV) for k, v in t.items()} for t in ts])
            loss = sum(losses.values()); opt.zero_grad(); loss.backward(); opt.step()
            for k in tot: tot[k] += losses[k].item()
            n += 1
        sched.step()
        mt = coco_metrics(*predict(m, va))
        hist.append(dict(epoch=ep + 1, loss_cls=tot["classification"] / n, loss_box=tot["bbox_regression"] / n,
                         val_map50=mt["map_50"], val_map=mt["map"]))
        print(f"[{name}] ep{ep+1:02d} cls {hist[-1]['loss_cls']:.3f} box {hist[-1]['loss_box']:.3f} "
              f"val mAP50 {mt['map_50']:.3f} mAP {mt['map']:.3f}")
        if mt["map_50"] > best: best, bad = mt["map_50"], 0; torch.save(m.state_dict(), f"best_{name}.pt")
        else:
            bad += 1
            if bad >= PROTOCOL["patience"]: print("рання зупинка"); break
    m.load_state_dict(torch.load(f"best_{name}.pt", map_location=DEV))
    h = pd.DataFrame(hist); h.to_csv(f"hist_{name}.csv", index=False); return m, h

# Навчання вже виконано в основному ноутбуці: завантажуємо збережені ваги та історії.
base = make_model(freeze_backbone=True); base.load_state_dict(torch.load("best_baseline_head.pt", map_location=DEV))
ft = make_model(); ft.load_state_dict(torch.load("best_retinanet_ft.pt", map_location=DEV))
h_base, h_det = pd.read_csv("hist_baseline_head.csv"), pd.read_csv("hist_retinanet_ft.csv")
print("val mAP50: лише голова =", h_base.val_map50.max(), "| донавчання всієї мережі =", h_det.val_map50.max())
det = base if h_base.val_map50.max() >= h_det.val_map50.max() else ft     # найкраща за валідацією
print("обрана модель:", "лише голова" if det is base else "донавчання")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for n, h in [("baseline (голова)", h_base), ("донавчання", h_det)]:
    l, = ax[0].plot(h.epoch, h.loss_cls, label=f"{n} cls"); ax[0].plot(h.epoch, h.loss_box, "--", color=l.get_color(), label=f"{n} box")
    ax[1].plot(h.epoch, h.val_map50, label=n)
ax[0].set(xlabel="епоха", ylabel="функція втрат (train)"); ax[0].legend(fontsize=8)
ax[1].set(xlabel="епоха", ylabel="mAP@0,5 (val)"); ax[1].legend()
plt.tight_layout(); plt.savefig("fig_curves.png", dpi=150); plt.show()

# %% [markdown]
# ## 4. Власне AP за протоколом COCO (лістинг 4.4), PR-криві, поріг за F1 на валідації

# %%
def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1]); ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.

def flat(P, G, c, size=None):
    preds = [(i, float(s), *b.tolist()) for i, p in enumerate(P)
             for b, l, s in zip(p["boxes"], p["labels"], p["scores"]) if l == c]
    gts = [(i, *b.tolist()) for i, g in enumerate(G) for b, l in zip(g["boxes"], g["labels"]) if l == c]
    if size:   # для AP_S/M/L: GT лише цього розміру; передбачення, зіставлені з GT іншого розміру, ігноруємо
        gts_all = gts; gts = [g for g in gts_all if size_group(g[1:]) == size]
        other = [g for g in gts_all if size_group(g[1:]) != size]
        preds = [p for p in preds if not any(o[0] == p[0] and iou(p[2:], o[1:]) >= .5 for o in other)
                 and (size_group(p[2:]) == size or any(g[0] == p[0] and iou(p[2:], g[1:]) >= .5 for g in gts))]
    return preds, gts

def pr_curve(preds, gts, thr=0.5):
    order = np.argsort([-p[1] for p in preds]); used = set(); tp = np.zeros(len(preds))
    for r, k in enumerate(order):
        img, _, *box = preds[k]; best, bi = 0., -1
        for j, g in enumerate(gts):
            if g[0] == img and j not in used:
                v = iou(box, g[1:])
                if v > best: best, bi = v, j
        if bi >= 0 and best >= thr: tp[r] = 1; used.add(bi)
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    rec = ctp / max(len(gts), 1); prec = ctp / np.maximum(ctp + cfp, 1e-9)
    scores = np.array([preds[k][1] for k in order]); return prec, rec, scores

def ap101(prec, rec):
    return sum((prec[rec >= r].max() if (rec >= r).any() else 0.) for r in np.linspace(0, 1, 101)) / 101

def ap(P, G, c, thr=0.5, size=None):
    pr, gt = flat(P, G, c, size); return ap101(*pr_curve(pr, gt, thr)[:2]) if gt else np.nan

def best_threshold(P, G):        # поріг за max F1 (усі класи разом)
    ts = np.arange(0.05, 0.96, 0.05); f = []
    for t in ts:
        tp = fp = n = 0
        for c in range(1, len(CLASSES) + 1):
            pr, gt = flat([{k: v[p["scores"] >= t] for k, v in p.items()} for p in P], G, c)
            prec, rec, _ = pr_curve(pr, gt)
            if len(prec): tp += rec[-1] * len(gt); fp += len(pr) - rec[-1] * len(gt)
            n += len(gt)
        P_, R_ = tp / max(tp + fp, 1e-9), tp / max(n, 1); f.append((t, P_, R_, 2 * P_ * R_ / max(P_ + R_, 1e-9)))
    f = pd.DataFrame(f, columns=["t", "precision", "recall", "F1"]); return f, f.loc[f.F1.idxmax()]

Pv, Gv = predict(det, dl("val")); f_tab, t_best = best_threshold(Pv, Gv)
print("поріг за max F1 на val:", dict(t_best.round(3)))
ax = f_tab.plot(x="t", y=["precision", "recall", "F1"]); ax.axvline(t_best.t, color="r")
plt.title("вибір порогу (val)"); plt.savefig("fig_threshold.png", dpi=150); plt.show()

# %% [markdown]
# ## 5. Тест: AP за класами, mAP@0,5, mAP@[0,5:0,95], AP за розмірами; порівняння з baseline

# %%
def report(m, factor=1, name=""):
    P, G = predict(m, dl("test", factor=factor))
    row = {"model": name, "factor": factor}
    for c in range(1, len(CLASSES) + 1):
        row[f"AP50_{CLASSES[c-1]}"] = ap(P, G, c, .5)
        row[f"AP5095_{CLASSES[c-1]}"] = np.nanmean([ap(P, G, c, t) for t in np.arange(.5, .96, .05)])
    row["mAP50"] = np.nanmean([row[f"AP50_{c}"] for c in CLASSES])
    row["mAP50_95"] = np.nanmean([row[f"AP5095_{c}"] for c in CLASSES])
    for s in "SML":
        row[f"AP50_{s}"] = np.nanmean([ap(P, G, c, .5, s) for c in range(1, len(CLASSES) + 1)])
    row["torchmetrics_map50"] = coco_metrics(P, G)["map_50"]   # контроль власної реалізації
    return row, P, G

r_base, _, _ = report(base, 1, "baseline (голова)")
r_ft, _, _ = report(ft, 1, "RetinaNet донавчання")
r_det, P_t, G_t = r_base, *report(base, 1, "")[1:]
r_det = dict(r_det); r_det["model"] = "обрана (лише голова)"
res = pd.DataFrame([r_base, r_ft]).set_index("model"); display(res.T.round(3)); res.to_csv("test_results.csv")

fig, ax = plt.subplots(figsize=(5, 4.5))
for c in range(1, len(CLASSES) + 1):
    prec, rec, _ = pr_curve(*flat(P_t, G_t, c))
    ax.plot(rec, prec, label=f"{CLASSES[c-1]}: AP={ap(P_t, G_t, c):.2f}")
ax.set(xlabel="повнота (recall)", ylabel="точність (precision)", title="PR-криві, IoU ≥ 0,5", xlim=(0, 1), ylim=(0, 1.02))
ax.legend(); plt.savefig("fig_pr.png", dpi=150); plt.show()

# %% [markdown]
# ## 6. Дослідження роздільної здатності: тест зі зменшенням у 2 рази (+ плитки)

# %%
r_x2, _, _ = report(det, 2, "обрана, зменшення x2")
gsd0 = ann[ann.split == "test"].gsd.median()
cmp_ = pd.DataFrame([r_det, r_x2]).set_index("model")
cmp_["GSD_emulated_m"] = [gsd0, gsd0 * 2]
display(cmp_[["GSD_emulated_m", "mAP50", "AP50_S", "AP50_M", "AP50_L"]].round(3)); cmp_.to_csv("resolution.csv")
ax = cmp_.set_index("GSD_emulated_m")[["mAP50", "AP50_S", "AP50_M", "AP50_L"]].plot(marker="o")
ax.set(xlabel="емульований GSD, м", ylabel="AP@0,5"); plt.savefig("fig_resolution.png", dpi=150); plt.show()

# %%
@torch.no_grad()
def detect_tiled(m, img_t, size=PROTOCOL["img_size"] // 2, overlap=0.2):
    """Плитки ½ кадру з перекриттям 20 %; кожну плитку модель масштабує до розміру входу (збільшення x2)."""
    m.eval(); _, H, W = img_t.shape; step = int(size * (1 - overlap))
    ys = sorted(set(range(0, max(H - size, 0), step)) | {max(H - size, 0)})
    xs = sorted(set(range(0, max(W - size, 0), step)) | {max(W - size, 0)})
    B, S, L = [], [], []
    for y0 in ys:
        for x0 in xs:
            o = m([img_t[:, y0:y0 + size, x0:x0 + size].to(DEV)])[0]
            B.append(o["boxes"].cpu() + torch.tensor([x0, y0, x0, y0])); S.append(o["scores"].cpu()); L.append(o["labels"].cpu())
    B, S, L = torch.cat(B), torch.cat(S), torch.cat(L)
    keep = torchvision.ops.batched_nms(B, S, L, PROTOCOL["nms_iou"])
    return {"boxes": B[keep], "scores": S[keep], "labels": L[keep]}

ds_t = DetDS(ann[ann.split == "test"]); P_tile, G_tile = [], []
for i in range(len(ds_t)):
    x, t = ds_t[i]; P_tile.append(detect_tiled(det, x)); G_tile.append(t)
print("цілий кадр  mAP50 =", round(r_det["mAP50"], 3))
print("плитки 1/2  mAP50 =", round(np.nanmean([ap(P_tile, G_tile, c) for c in range(1, len(CLASSES) + 1)]), 3),
      "| AP50_S =", round(np.nanmean([ap(P_tile, G_tile, c, .5, "S") for c in range(1, len(CLASSES) + 1)]), 3))

# %% [markdown]
# ## 7. Аналіз помилок за категоріями TIDE при обраному порозі

# %%
def categorize(P, G, t):
    cats = dict(Cls=0, Loc=0, Both=0, Dupe=0, Bkg=0, Miss=0); miss_by_size = {"S": [0, 0], "M": [0, 0], "L": [0, 0]}
    examples = {k: [] for k in cats}; TP = 0
    for i, (p, g) in enumerate(zip(P, G)):
        keep = p["scores"] >= t; pb, pl, ps = p["boxes"][keep].tolist(), p["labels"][keep].tolist(), p["scores"][keep].tolist()
        gb, gl = g["boxes"].tolist(), g["labels"].tolist(); matched = set()
        for k in np.argsort(ps)[::-1]:
            ious = [iou(pb[k], b) for b in gb]
            same = [(v, j) for j, v in enumerate(ious) if gl[j] == pl[k]]
            best_s = max(same) if same else (0, -1); best_any = max(ious) if ious else 0
            if best_s[0] >= .5 and best_s[1] not in matched: TP += 1; matched.add(best_s[1]); continue
            if best_s[0] >= .5: c = "Dupe"
            elif best_any >= .5: c = "Cls"
            elif best_s[0] >= .1: c = "Loc"
            elif best_any >= .1: c = "Both"
            else: c = "Bkg"
            cats[c] += 1; examples[c].append(i)
        for j, b in enumerate(gb):
            s = size_group(b); miss_by_size[s][1] += 1
            if j not in matched: miss_by_size[s][0] += 1
        n_miss = len(gb) - len(matched); cats["Miss"] += n_miss
        if n_miss: examples["Miss"].append(i)
    return TP, cats, miss_by_size, examples

TP, cats, mbs, ex = categorize(P_t, G_t, t_best.t)
print("TP =", TP, cats)
plt.bar(list(cats), list(cats.values()), color="grey"); plt.title(f"Категорії помилок при порозі {t_best.t:.2f} (TP = {TP})")
for k, v in cats.items(): plt.text(list(cats).index(k), v, v, ha="center", va="bottom")
plt.savefig("fig_errors_cat.png", dpi=150); plt.show()
miss_tab = pd.DataFrame({s: dict(missed=v[0], total=v[1], share=v[0] / max(v[1], 1)) for s, v in mbs.items()}).T
display(miss_tab.round(3)); miss_tab.to_csv("misses_by_size.csv")

# %%
def show_examples(cat, n=4):
    idx = list(dict.fromkeys(ex[cat]))[:n]
    if not idx: print("немає прикладів", cat); return
    test_rows = ann[ann.split == "test"].reset_index(drop=True)
    fig, ax = plt.subplots(1, len(idx), figsize=(5 * len(idx), 5))
    for a, i in zip(np.atleast_1d(ax), idx):
        im = cv2.cvtColor(cv2.imread(test_rows.file[i]), cv2.COLOR_BGR2RGB)
        im = draw(im, G_t[i]["boxes"].tolist(), G_t[i]["labels"].tolist(), (0, 255, 0))
        k = P_t[i]["scores"] >= t_best.t
        im = draw(im, P_t[i]["boxes"][k].tolist(), P_t[i]["labels"][k].tolist(), (255, 0, 0), P_t[i]["scores"][k].tolist())
        a.imshow(im); a.set_title(f"{cat}: {test_rows.event[i]}"); a.axis("off")
    plt.suptitle("зелені — істинні рамки, червоні — передбачення"); plt.tight_layout()
    plt.savefig(f"fig_ex_{cat}.png", dpi=110); plt.show()
for c in ["Miss", "Loc", "Cls", "Bkg", "Dupe"]: show_examples(c)
