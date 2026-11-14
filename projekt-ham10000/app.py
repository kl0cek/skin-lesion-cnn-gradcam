"""
Aplikacja Gradio: klasyfikacja zmiany skórnej + mapa Grad-CAM.
"""
import argparse
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd

from model import CLASSES, NAMES, gradcam, load_image, load_model, overlay, to_tensor

ROOT = Path(__file__).parent
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default=ROOT / "out" / "models" / "E5_s0.pt")
args = parser.parse_args()

model = load_model(args.checkpoint)
EXPLAIN = ["przewidywana klasa"] + [NAMES[c] for c in CLASSES]


def predict(img, explain):
    if img is None:
        return None, None
    arr = load_image(img)
    class_idx = None if explain == EXPLAIN[0] else EXPLAIN.index(explain) - 1
    cam, probs, _ = gradcam(model, to_tensor(arr), class_idx)
    scores = {NAMES[c]: float(p) for c, p in zip(CLASSES, probs)}
    return scores, (overlay(arr, cam) * 255).astype(np.uint8)


def examples():
    # po jednym zdjęciu z każdej klasy ze zbioru walidacyjnego (testu nie używamy)
    splits = ROOT / "data" / "splits.csv"
    if not splits.exists():
        return []
    paths = {p.stem: p for p in (ROOT / "data").rglob("ISIC_*.jpg")}
    val = pd.read_csv(splits).query("split == 'val'")
    ids = [(c, val[val.dx == c].image_id.iloc[0]) for c in CLASSES]
    return [(str(paths[i]), f"prawdziwa klasa: {NAMES[c]}") for c, i in ids if i in paths]


with gr.Blocks(title="Klasyfikacja zmian skórnych") as demo:
    gr.Markdown(
        "# Klasyfikacja zmian skórnych\n"
        "EfficientNet-B0 uczony na zbiorze HAM10000 (7 klas). Wgraj zdjęcie dermatoskopowe – model wskaże "
        "najbardziej prawdopodobne typy zmiany, a mapa Grad-CAM pokaże, które fragmenty zdjęcia wpłynęły na decyzję."
    )
    with gr.Row():
        with gr.Column():
            image = gr.Image(type="pil", label="Zdjęcie dermatoskopowe")
            explain = gr.Dropdown(EXPLAIN, value=EXPLAIN[0], label="Mapa Grad-CAM dla klasy")
            button = gr.Button("Klasyfikuj", variant="primary")
        with gr.Column():
            result = gr.Label(num_top_classes=3, label="Wynik")
            heatmap = gr.Image(label="Grad-CAM – na co patrzył model")
    gr.Markdown(
        "**Projekt edukacyjny – nie jest narzędziem diagnostycznym.** Model uczono wyłącznie na zdjęciach "
        "dermatoskopowych; dla zwykłych zdjęć z telefonu wyniki mogą być błędne. Model zawsze wskazuje jedną "
        "z 7 klas, także dla zdjęć, które nie przedstawiają zmiany skórnej."
    )
    if ex := examples():
        gr.Examples([[p] for p, _ in ex], inputs=image, example_labels=[label for _, label in ex],
                    label="Przykłady ze zbioru walidacyjnego")

    button.click(predict, [image, explain], [result, heatmap], api_name="predict")
    explain.change(predict, [image, explain], [result, heatmap])

if __name__ == "__main__":
    demo.launch()
