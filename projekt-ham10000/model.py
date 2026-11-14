"""Wspólny kod dla kamienia 3 i aplikacji: model, przetwarzanie zdjęć i Grad-CAM.

Przetwarzanie musi być identyczne jak w kamien2.ipynb (resize do 256x256, potem 224 i normalizacja ImageNet).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models
from torchvision.transforms import v2

CLASSES = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]
NAMES = {
    "nv": "znamię barwnikowe (nv)",
    "mel": "czerniak (mel)",
    "bkl": "łagodne rogowacenie (bkl)",
    "bcc": "rak podstawnokomórkowy (bcc)",
    "akiec": "rogowacenie słoneczne (akiec)",
    "vasc": "zmiana naczyniowa (vasc)",
    "df": "włókniak skóry (df)",
}

EVAL_TF = v2.Compose([
    v2.Resize(224),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class SimpleCNN(nn.Module):
    # baseline: 4 bloki conv-bn-relu-pool, uczony od zera
    def __init__(self, n_classes=7):
        super().__init__()
        layers, c_in = [], 3
        for c_out in [32, 64, 128, 256]:
            layers += [nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(), nn.MaxPool2d(2)]
            c_in = c_out
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


def build_model(name, pretrained=True):
    if name == "cnn":
        return SimpleCNN()
    if name == "resnet50":
        m = models.resnet50(weights="IMAGENET1K_V2" if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, len(CLASSES))
        return m
    if name == "efficientnet_b0":
        m = models.efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, len(CLASSES))
        return m
    raise ValueError(name)


def load_model(path, name="efficientnet_b0", device="cpu"):
    model = build_model(name, pretrained=False)  # wagi i tak nadpisujemy checkpointem
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()


def target_layer(model):
    # ostatnia warstwa konwolucyjna – z niej Grad-CAM bierze mapę cech
    if isinstance(model, SimpleCNN):
        return model.features[-2]
    if isinstance(model, models.ResNet):
        return model.layer4[-1]
    return model.features[-1]  # EfficientNet


def load_image(img):
    """PIL / ścieżka -> uint8 256x256x3, dokładnie jak przy treningu."""
    if not isinstance(img, Image.Image):
        img = Image.open(img)
    return np.array(img.convert("RGB").resize((256, 256)))


def to_tensor(arr):
    return EVAL_TF(torch.from_numpy(arr).permute(2, 0, 1)).unsqueeze(0)  # 1x3x224x224


def gradcam(model, x, class_idx=None):
    """Zwraca (mapa 224x224 w [0, 1], prawdopodobieństwa, wyjaśniana klasa)."""
    store = {}

    def hook(_, __, out):
        store["act"] = out
        out.register_hook(lambda g: store.__setitem__("grad", g))

    h = target_layer(model).register_forward_hook(hook)
    try:
        with torch.enable_grad():
            logits = model(x)
            probs = logits.softmax(1)[0].detach()
            c = int(probs.argmax()) if class_idx is None else class_idx
            model.zero_grad()
            logits[0, c].backward()
    finally:
        h.remove()

    # waga kanału = średni gradient; mapa = ważona suma kanałów, tylko wpływ dodatni (ReLU)
    weights = store["grad"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * store["act"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam / (cam.max() + 1e-8)
    return cam.detach().cpu().numpy(), probs.cpu().numpy(), c


def overlay(arr, cam, alpha=0.6):
    """Nakłada mapę na zdjęcie; miejsca o małym znaczeniu zostają nieprzyciemnione."""
    import matplotlib

    img = np.asarray(Image.fromarray(arr).resize(cam.shape[::-1])).astype(np.float32) / 255
    heat = matplotlib.colormaps["inferno"](cam)[..., :3]
    a = alpha * cam[..., None]
    return ((1 - a) * img + a * heat).clip(0, 1)
