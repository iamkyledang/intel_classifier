# Intel Scene Classifier — Web Demo

A static website that classifies a photo into one of six Intel Scene
categories (`buildings`, `forest`, `glacier`, `mountain`, `sea`, `street`)
using a model trained in [../main.ipynb](../main.ipynb):

- **CNN** — PyTorch model exported to [ONNX](https://onnx.ai/) and run in
  the browser with [onnxruntime-web](https://onnxruntime.ai/docs/tutorials/web/).
- **Classical (SVM)** — Color-histogram + edge/Sobel + HOG features,
  reimplemented in JavaScript, feeding the real trained `cv2.ml.SVM`.

Everything runs **entirely in the browser** — no backend server, no image
uploads — so it can also be hosted for free on GitHub Pages.

## Files

- `index.html`, `style.css` — page and styling
- `app.js` — inference logic (preprocessing, CNN + SVM)
- `export_models.py` — generates the `models/` folder
- `models/` — exported model assets used by the page

## Generate the model assets

Requires `cnn_scene_final.pt` and `svm_scene_final.pt` from the notebook
(in the parent `CNN/` folder). Run:

```powershell
python image_classify/export_models.py
```

This writes `model.onnx`, `cnn_config.json`, `support_vectors.bin`, and
`svm_config.json` into `models/`. Re-run after retraining.

## Run locally

Browsers block loading local files directly, so serve the folder:

```powershell
python -m http.server 8000
```

Then open <http://localhost:8000/>.

## Deploy to GitHub Pages

1. Push the repo to GitHub (including `models/`).
2. Go to **Settings → Pages**.
3. Set **Source** to **Deploy from a branch**, pick your branch and
   **/ (root)**.
4. Your site will be live at:

   ```
   https://<your-username>.github.io/<your-repo-name>/
   ```

## How it works

1. Pick an image and a model, then click **Classify image**.
2. The image is resized to the model's expected input size on an
   off-screen canvas.
3. **CNN:** pixels are normalized and run through the ONNX model via
   `onnxruntime-web`; output logits go through softmax.
4. **SVM:** HSV color histogram, Canny edge density, Sobel gradient
   histogram, and HOG features are computed in JavaScript, scaled, then
   scored against the support vectors with the RBF kernel using OpenCV's
   one-vs-one voting scheme.

## Caveats

- **CNN** results should closely match the notebook's reported accuracy —
  same weights, just running through ONNX.
- **SVM** feature extraction is a JavaScript reimplementation of OpenCV's
  Canny/HOG (the learned decision boundary itself is exact), so
  predictions are usually sensible but may not always exactly match the
  original Python pipeline.
- Large photos are automatically downscaled before processing.
