# Розширений аналіз помилок: помилки + найменш упевнені правильні прогнози, Grad-CAM для кожного класу.
# Виконується після основного ноутбука (використовує best_vit_finetune.pt і cls_split.csv).
import numpy as np, pandas as pd, torch, torch.nn.functional as F, timm, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms as T
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

CLASSES = ["mineral", "wood", "metal", "mixed", "background"]; S = 224
df = pd.read_csv("cls_split.csv"); test_df = df[df.split == "test"].reset_index(drop=True)
m = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=len(CLASSES))
m.load_state_dict(torch.load("best_vit_finetune.pt", map_location="cpu")); m.eval()
tf = T.Compose([T.Resize((S, S)), T.ToTensor(), T.Normalize([.5] * 3, [.5] * 3)])
with torch.no_grad():
    pr = torch.cat([F.softmax(m(torch.stack([tf(Image.open(p).convert("RGB")) for p in test_df.path[i:i + 16]])), 1)
                    for i in range(0, len(test_df), 16)]).numpy()
y = test_df.label.map(CLASSES.index).values; p = pr.argmax(1); conf = pr.max(1)
err = np.where(y != p)[0]; ok = np.where(y == p)[0]
show = list(err) + list(ok[np.argsort(conf[ok])][:8 - len(err)])
fig, ax = plt.subplots(1, 8, figsize=(20, 3.4))
for a, k in zip(ax, show):
    a.imshow(Image.open(test_df.path[k])); a.axis("off")
    tag = "ПОМИЛКА" if y[k] != p[k] else "вірно"
    a.set_title(f"{tag}: {CLASSES[y[k]]}→{CLASSES[p[k]]}\np={conf[k]:.2f}, {test_df.scene[k]}", fontsize=8,
                color="red" if y[k] != p[k] else "black")
plt.suptitle("Помилки та найменш упевнені правильні прогнози (тест)"); plt.tight_layout()
plt.savefig("fig_errors.png", dpi=130)
print("помилок:", len(err), "| мінімальна впевненість серед правильних:", round(float(conf[ok].min()), 3))
print("середня впевненість за класами:", {c: round(float(conf[y == i].mean()), 3) for i, c in enumerate(CLASSES)})

for prm in m.parameters(): prm.requires_grad = True
def reshape_vit(t, h=14, w=14):
    r = t[:, 1:, :].reshape(t.size(0), h, w, t.size(2)); return r.permute(0, 3, 1, 2)
cam = GradCAM(model=m, target_layers=[m.blocks[-1].norm1], reshape_transform=reshape_vit)
pick = list(err) + [ok[y[ok] == i][0] for i in range(len(CLASSES))]
fig, ax = plt.subplots(1, len(pick), figsize=(3.3 * len(pick), 3.6))
for a, k in zip(ax, pick):
    img = Image.open(test_df.path[k]).convert("RGB").resize((S, S))
    g = cam(input_tensor=tf(img)[None], targets=[ClassifierOutputTarget(int(p[k]))])[0]
    a.imshow(show_cam_on_image(np.asarray(img) / 255., g, use_rgb=True)); a.axis("off")
    a.set_title(f"{CLASSES[y[k]]}→{CLASSES[p[k]]}", fontsize=9, color="red" if y[k] != p[k] else "black")
plt.suptitle("Grad-CAM (останній блок ViT): помилка та по одному правильному прикладу кожного класу")
plt.tight_layout(); plt.savefig("fig_gradcam.png", dpi=130)
print("Grad-CAM збережено для", len(pick), "прикладів")
