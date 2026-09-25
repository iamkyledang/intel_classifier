"""
Export the trained CNN and SVM pipelines (produced by ../main.ipynb) into
web-friendly assets that the static site in this folder can load directly
in the browser — no Python server required.

Run this once (from the CNN folder or anywhere) after the notebook has
produced `cnn_scene_final.pt` and `svm_scene_final.pt` in the CNN folder:

    python image_classify/export_models.py

Outputs (written to ./image_classify/models/):
    model.onnx           - CNN exported to ONNX (run in-browser via onnxruntime-web)
    cnn_config.json      - class list, image size, normalization mean/std for the CNN
    support_vectors.bin  - the trained SVM's support vectors, as raw row-major
                           float32 binary (sv_total x var_count) for fast loading
    svm_config.json      - everything else needed to reproduce cv2.ml.SVM.predict()
                           in plain JavaScript: gamma, per-class-pair decision
                           functions (rho/alpha/support-vector-indices), the
                           feature scaler (mean/std), HOG params, image size and
                           class names. Decision functions are read out via
                           OpenCV's own Python API (getDecisionFunction), which
                           is validated against cv2.ml.SVM.predict() to guarantee
                           the browser-side voting math is exactly equivalent.
"""

import json
import os

import cv2
import numpy as np
import torch
import torch.nn as nn

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CNN_DIR = os.path.dirname(SCRIPT_DIR)  # the "CNN" folder, parent of image_classify

CNN_CHECKPOINT_PATH = os.path.join(CNN_DIR, "cnn_scene_final.pt")
SVM_CHECKPOINT_PATH = os.path.join(CNN_DIR, "svm_scene_final.pt")

OUT_DIR = os.path.join(SCRIPT_DIR, "models")
os.makedirs(OUT_DIR, exist_ok=True)


# ── CNN architecture (must match main.ipynb exactly) ─────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool=False):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class SceneCNN(nn.Module):
    def __init__(self, num_classes=6, dropout_p=0.5):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32, pool=True),
            ConvBlock(32, 64, pool=True),
            ConvBlock(64, 128, pool=True),
            ConvBlock(128, 256, pool=True),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def export_cnn():
    print(f"Loading CNN checkpoint from '{CNN_CHECKPOINT_PATH}'...")
    checkpoint = torch.load(CNN_CHECKPOINT_PATH, map_location="cpu")

    classes = checkpoint["classes"]
    img_size = checkpoint["img_size"]
    mean = checkpoint["mean"]
    std = checkpoint["std"]

    model = SceneCNN(num_classes=len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    onnx_path = os.path.join(OUT_DIR, "model.onnx")
    dummy_input = torch.zeros(1, 3, img_size, img_size)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=13,
    )
    print(f"  -> Saved ONNX model to '{onnx_path}'")

    cnn_config = {
        "classes": classes,
        "img_size": img_size,
        "mean": mean,
        "std": std,
    }
    cnn_config_path = os.path.join(OUT_DIR, "cnn_config.json")
    with open(cnn_config_path, "w") as f:
        json.dump(cnn_config, f, indent=2)
    print(f"  -> Saved CNN config to '{cnn_config_path}'")


def export_svm():
    print(f"Loading SVM checkpoint from '{SVM_CHECKPOINT_PATH}'...")
    checkpoint = torch.load(SVM_CHECKPOINT_PATH, map_location="cpu")
    classes = checkpoint["classes"]
    num_classes = len(classes)

    # Reconstruct the live cv2.ml.SVM object from the stored XML so we can
    # query it through OpenCV's own (trusted) API rather than hand-parsing XML.
    tmp_xml_path = os.path.join(OUT_DIR, "_svm_tmp_load.xml")
    with open(tmp_xml_path, "w") as f:
        f.write(checkpoint["svm_xml"])
    svm = cv2.ml.SVM_load(tmp_xml_path)
    os.remove(tmp_xml_path)

    var_count = svm.getVarCount()
    support_vectors = svm.getSupportVectors().astype(np.float32)
    sv_total = support_vectors.shape[0]
    gamma = svm.getGamma()

    # One-vs-one decision functions, in the standard (i, j) i<j pair order —
    # this matches how OpenCV indexes getDecisionFunction() for a C_SVC model.
    decision_functions = []
    pair_index = 0
    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            rho, alpha, svidx = svm.getDecisionFunction(pair_index)
            decision_functions.append({
                "i": i,
                "j": j,
                "rho": float(rho),
                "alpha": np.asarray(alpha).flatten().astype(np.float64).tolist(),
                "svidx": np.asarray(svidx).flatten().astype(np.int32).tolist(),
            })
            pair_index += 1

    sv_bin_path = os.path.join(OUT_DIR, "support_vectors.bin")
    support_vectors.tofile(sv_bin_path)
    print(f"  -> Saved {sv_total} support vectors ({var_count} dims) to '{sv_bin_path}'")

    svm_config = {
        "var_count": int(var_count),
        "sv_total": int(sv_total),
        "gamma": float(gamma),
        "classes": classes,
        "img_size": checkpoint["img_size"],
        "hog_params": checkpoint["hog_params"],
        "feat_mean": checkpoint["feat_mean"].tolist(),
        "feat_std": checkpoint["feat_std"].tolist(),
        "decision_functions": decision_functions,
    }
    svm_config_path = os.path.join(OUT_DIR, "svm_config.json")
    with open(svm_config_path, "w") as f:
        json.dump(svm_config, f)
    print(f"  -> Saved SVM config to '{svm_config_path}'")


if __name__ == "__main__":
    export_cnn()
    export_svm()
    print("\nDone. Web assets are in:", OUT_DIR)
