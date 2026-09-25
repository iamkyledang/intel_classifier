"use strict";

/*
 * Intel Scene Classifier — client-side inference for two models:
 *   1. CNN   — exact trained model, run via ONNX Runtime Web.
 *   2. SVM   — classical Color-Histogram + Edge/Sobel + HOG features,
 *              reimplemented in plain JavaScript, fed into the *real*
 *              trained SVM (support vectors + one-vs-one decision
 *              functions extracted from the notebook's cv2.ml.SVM model).
 *
 * Everything runs locally in the browser. Model assets are fetched from
 * ./models/ — see export_models.py for how they were produced.
 */

// ── DOM references ───────────────────────────────────────────────────────────
const fileInput = document.getElementById("fileInput");
const dropzone = document.getElementById("dropzone");
const dropzoneText = document.getElementById("dropzoneText");
const preview = document.getElementById("preview");
const classifyBtn = document.getElementById("classifyBtn");
const statusEl = document.getElementById("status");
const resultPanel = document.getElementById("resultPanel");
const resultLabel = document.getElementById("resultLabel");
const resultConfidence = document.getElementById("resultConfidence");
const resultBars = document.getElementById("resultBars");
const timingEl = document.getElementById("timing");

let currentImage = null; // HTMLImageElement of the currently loaded photo

// ── Lazy-loaded model assets (fetched once, cached) ──────────────────────────
let cnnConfigPromise = null;
let cnnSessionPromise = null;
let svmConfigPromise = null;
let svmSupportVectorsPromise = null;

function getCnnConfig() {
  if (!cnnConfigPromise) {
    cnnConfigPromise = fetch("models/cnn_config.json").then((r) => r.json());
  }
  return cnnConfigPromise;
}

function getCnnSession() {
  if (!cnnSessionPromise) {
    cnnSessionPromise = ort.InferenceSession.create("models/model.onnx");
  }
  return cnnSessionPromise;
}

function getSvmConfig() {
  if (!svmConfigPromise) {
    svmConfigPromise = fetch("models/svm_config.json").then((r) => r.json());
  }
  return svmConfigPromise;
}

function getSvmSupportVectors() {
  if (!svmSupportVectorsPromise) {
    svmSupportVectorsPromise = fetch("models/support_vectors.bin")
      .then((r) => r.arrayBuffer())
      .then((buf) => new Float32Array(buf));
  }
  return svmSupportVectorsPromise;
}

// ── Image loading / drag & drop ──────────────────────────────────────────────
// Note: dropzone is a <label for="fileInput">, so clicking it already opens
// the file picker natively — no extra click() call needed here.

dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("dragover");
});
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
  const file = e.dataTransfer.files && e.dataTransfer.files[0];
  if (file) loadFile(file);
});

fileInput.addEventListener("change", () => {
  const file = fileInput.files && fileInput.files[0];
  if (file) loadFile(file);
});

function loadFile(file) {
  if (!file.type.startsWith("image/")) {
    setStatus("Please choose an image file.", true);
    return;
  }
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onload = () => {
    currentImage = img;
    preview.src = url;
    preview.hidden = false;
    dropzoneText.hidden = true;
    classifyBtn.disabled = false;
    resultLabel.textContent = "No image classified yet";
    resultConfidence.textContent = "";
    timingEl.textContent = "";
    renderBars(SCENE_CLASSES, null);
    setStatus("");
  };
  img.onerror = () => setStatus("Could not load that image.", true);
  img.src = url;
}

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.classList.toggle("error", isError);
}

// ── Shared image preprocessing ───────────────────────────────────────────────
function resizeImageToImageData(img, size) {
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(img, 0, 0, size, size);
  return ctx.getImageData(0, 0, size, size);
}

function softmax(logits) {
  const max = Math.max(...logits);
  const exps = logits.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

// ── CNN inference (ONNX Runtime Web) ─────────────────────────────────────────
async function runCnn(img) {
  const cfg = await getCnnConfig();
  const size = cfg.img_size;
  const { data } = resizeImageToImageData(img, size);

  const plane = size * size;
  const chw = new Float32Array(3 * plane);
  for (let p = 0; p < plane; p++) {
    const r = data[p * 4] / 255;
    const g = data[p * 4 + 1] / 255;
    const b = data[p * 4 + 2] / 255;
    chw[p] = (r - cfg.mean[0]) / cfg.std[0];
    chw[plane + p] = (g - cfg.mean[1]) / cfg.std[1];
    chw[2 * plane + p] = (b - cfg.mean[2]) / cfg.std[2];
  }

  const session = await getCnnSession();
  const tensor = new ort.Tensor("float32", chw, [1, 3, size, size]);
  const feeds = { [session.inputNames[0]]: tensor };
  const outputs = await session.run(feeds);
  const logits = Array.from(outputs[session.outputNames[0]].data);
  const probs = softmax(logits);

  return { classes: cfg.classes, probs };
}

// ── Classical feature extraction (plain JS re-implementation) ───────────────
function toGrayscale(imageData) {
  const { data, width, height } = imageData;
  const gray = new Float32Array(width * height);
  for (let i = 0, p = 0; i < data.length; i += 4, p++) {
    gray[p] = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
  }
  return { gray, width, height };
}

// OpenCV's 8-bit HSV scaling: H in [0,180), S in [0,255], V in [0,255]
function rgbToOpenCvHsv(r, g, b) {
  const rf = r / 255, gf = g / 255, bf = b / 255;
  const max = Math.max(rf, gf, bf);
  const min = Math.min(rf, gf, bf);
  const delta = max - min;
  const v = max;
  const s = max === 0 ? 0 : delta / max;
  let h = 0;
  if (delta !== 0) {
    if (max === rf) h = 60 * (((gf - bf) / delta) % 6);
    else if (max === gf) h = 60 * ((bf - rf) / delta + 2);
    else h = 60 * ((rf - gf) / delta + 4);
  }
  if (h < 0) h += 360;
  return [h / 2, s * 255, v * 255];
}

function colorHistogram(imageData, bins = [8, 8, 8]) {
  const { data } = imageData;
  const [hBins, sBins, vBins] = bins;
  const hist = new Float64Array(hBins * sBins * vBins);
  for (let i = 0; i < data.length; i += 4) {
    const [h, s, v] = rgbToOpenCvHsv(data[i], data[i + 1], data[i + 2]);
    let hb = Math.min(hBins - 1, Math.floor((h * hBins) / 180));
    let sb = Math.min(sBins - 1, Math.floor((s * sBins) / 256));
    let vb = Math.min(vBins - 1, Math.floor((v * vBins) / 256));
    hist[hb * sBins * vBins + sb * vBins + vb] += 1;
  }
  let norm = 0;
  for (let i = 0; i < hist.length; i++) norm += hist[i] * hist[i];
  norm = Math.sqrt(norm);
  const out = new Float64Array(hist.length);
  if (norm > 0) for (let i = 0; i < hist.length; i++) out[i] = hist[i] / norm;
  return out;
}

// 3x3 Sobel gradients with edge-replicated borders (matches cv2.Sobel ksize=3)
function sobelGradients(gray, width, height) {
  const gx = new Float32Array(width * height);
  const gy = new Float32Array(width * height);
  const at = (x, y) => {
    const cx = x < 0 ? 0 : x >= width ? width - 1 : x;
    const cy = y < 0 ? 0 : y >= height ? height - 1 : y;
    return gray[cy * width + cx];
  };
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      gx[idx] =
        -at(x - 1, y - 1) + at(x + 1, y - 1) +
        -2 * at(x - 1, y) + 2 * at(x + 1, y) +
        -at(x - 1, y + 1) + at(x + 1, y + 1);
      gy[idx] =
        -at(x - 1, y - 1) - 2 * at(x, y - 1) - at(x + 1, y - 1) +
        at(x - 1, y + 1) + 2 * at(x, y + 1) + at(x + 1, y + 1);
    }
  }
  return { gx, gy };
}

// Canny edge detector (default OpenCV behaviour: L2gradient=False, aperture=3)
// implemented from the Sobel gradients above, so it matches cv2.Canny(gray, low, high).
function cannyEdgeDensity(gx, gy, width, height, lowThresh, highThresh) {
  const n = width * height;
  const mag = new Float32Array(n);
  const dir = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    mag[i] = Math.abs(gx[i]) + Math.abs(gy[i]);
    let angle = (Math.atan2(gy[i], gx[i]) * 180) / Math.PI;
    if (angle < 0) angle += 180;
    if (angle < 22.5 || angle >= 157.5) dir[i] = 0;
    else if (angle < 67.5) dir[i] = 1;
    else if (angle < 112.5) dir[i] = 2;
    else dir[i] = 3;
  }

  const at = (x, y) => (x < 0 || x >= width || y < 0 || y >= height ? 0 : mag[y * width + x]);
  const suppressed = new Float32Array(n);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      const m = mag[idx];
      let n1, n2;
      if (dir[idx] === 0) { n1 = at(x - 1, y); n2 = at(x + 1, y); }
      else if (dir[idx] === 1) { n1 = at(x - 1, y + 1); n2 = at(x + 1, y - 1); }
      else if (dir[idx] === 2) { n1 = at(x, y - 1); n2 = at(x, y + 1); }
      else { n1 = at(x - 1, y - 1); n2 = at(x + 1, y + 1); }
      suppressed[idx] = m >= n1 && m >= n2 ? m : 0;
    }
  }

  const STRONG = 2, WEAK = 1;
  const state = new Uint8Array(n);
  const stack = [];
  for (let i = 0; i < n; i++) {
    if (suppressed[i] >= highThresh) { state[i] = STRONG; stack.push(i); }
    else if (suppressed[i] >= lowThresh) { state[i] = WEAK; }
  }
  while (stack.length) {
    const idx = stack.pop();
    const x = idx % width, y = (idx / width) | 0;
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        if (dx === 0 && dy === 0) continue;
        const nx = x + dx, ny = y + dy;
        if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
        const nIdx = ny * width + nx;
        if (state[nIdx] === WEAK) { state[nIdx] = STRONG; stack.push(nIdx); }
      }
    }
  }

  let edgeCount = 0;
  for (let i = 0; i < n; i++) if (state[i] === STRONG) edgeCount++;
  return edgeCount / n;
}

function gradientMagnitudeHistogram(gx, gy, bins = 16, rangeMax = 255) {
  const n = gx.length;
  const hist = new Float64Array(bins);
  const binWidth = rangeMax / bins;
  let total = 0;
  for (let i = 0; i < n; i++) {
    const m = Math.sqrt(gx[i] * gx[i] + gy[i] * gy[i]);
    if (m >= 0 && m <= rangeMax) {
      const b = Math.min(bins - 1, Math.floor(m / binWidth));
      hist[b] += 1;
      total += 1;
    }
  }
  const denom = total + 1e-7;
  const out = new Float64Array(bins);
  for (let i = 0; i < bins; i++) out[i] = hist[i] / denom;
  return out;
}

// Simplified HOG descriptor (Dalal-Triggs style: per-cell histograms with
// bilinear orientation interpolation, then per-block L2-Hys normalization).
// Approximates cv2.HOGDescriptor.compute() — see README for caveats.
function computeHog(gray, width, height, params) {
  const [winW, winH] = params._winSize;
  const [blockW, blockH] = params._blockSize;
  const [strideX, strideY] = params._blockStride;
  const [cellW, cellH] = params._cellSize;
  const nbins = params._nbins;

  const at = (x, y) => {
    const cx = x < 0 ? 0 : x >= width ? width - 1 : x;
    const cy = y < 0 ? 0 : y >= height ? height - 1 : y;
    return gray[cy * width + cx];
  };

  const n = width * height;
  const mag = new Float32Array(n);
  const angle = new Float32Array(n);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const gx = at(x + 1, y) - at(x - 1, y);
      const gy = at(x, y + 1) - at(x, y - 1);
      const idx = y * width + x;
      mag[idx] = Math.sqrt(gx * gx + gy * gy);
      let a = (Math.atan2(gy, gx) * 180) / Math.PI;
      if (a < 0) a += 180;
      if (a >= 180) a -= 180;
      angle[idx] = a;
    }
  }

  const binWidth = 180 / nbins;
  function cellHistogram(x0, y0) {
    const hist = new Float64Array(nbins);
    for (let y = y0; y < y0 + cellH; y++) {
      for (let x = x0; x < x0 + cellW; x++) {
        const idx = y * width + x;
        const m = mag[idx];
        const bin = angle[idx] / binWidth - 0.5;
        const b0 = Math.floor(bin);
        const frac = bin - b0;
        const nb0 = ((b0 % nbins) + nbins) % nbins;
        const nb1 = (nb0 + 1) % nbins;
        hist[nb0] += m * (1 - frac);
        hist[nb1] += m * frac;
      }
    }
    return hist;
  }

  const numCellsX = winW / cellW;
  const numCellsY = winH / cellH;
  const cellHists = [];
  for (let cy = 0; cy < numCellsY; cy++) {
    const row = [];
    for (let cx = 0; cx < numCellsX; cx++) row.push(cellHistogram(cx * cellW, cy * cellH));
    cellHists.push(row);
  }

  const cellsPerBlockX = blockW / cellW;
  const cellsPerBlockY = blockH / cellH;
  const numBlocksX = Math.floor((winW - blockW) / strideX) + 1;
  const numBlocksY = Math.floor((winH - blockH) / strideY) + 1;

  const features = [];
  for (let by = 0; by < numBlocksY; by++) {
    for (let bx = 0; bx < numBlocksX; bx++) {
      const cellX0 = (bx * strideX) / cellW;
      const cellY0 = (by * strideY) / cellH;
      const block = [];
      for (let cy = 0; cy < cellsPerBlockY; cy++) {
        for (let cx = 0; cx < cellsPerBlockX; cx++) {
          const hist = cellHists[cellY0 + cy][cellX0 + cx];
          for (let b = 0; b < nbins; b++) block.push(hist[b]);
        }
      }
      let norm = Math.sqrt(block.reduce((a, v) => a + v * v, 0) + 1e-12);
      for (let i = 0; i < block.length; i++) block[i] /= norm;
      for (let i = 0; i < block.length; i++) if (block[i] > 0.2) block[i] = 0.2;
      norm = Math.sqrt(block.reduce((a, v) => a + v * v, 0) + 1e-12);
      for (let i = 0; i < block.length; i++) block[i] /= norm;
      features.push(...block);
    }
  }
  return features;
}

// ── SVM inference (RBF one-vs-one voting, exactly matching cv2.ml.SVM) ──────
async function runSvm(img) {
  const cfg = await getSvmConfig();
  const supportVectors = await getSvmSupportVectors();

  const size = cfg.img_size;
  const imageData = resizeImageToImageData(img, size);
  const { gray, width, height } = toGrayscale(imageData);

  const colorFeat = colorHistogram(imageData);
  const { gx, gy } = sobelGradients(gray, width, height);
  const edgeDensity = cannyEdgeDensity(gx, gy, width, height, 100, 200);
  const gradHist = gradientMagnitudeHistogram(gx, gy);
  const hogFeat = computeHog(gray, width, height, cfg.hog_params);

  const raw = new Float64Array(cfg.var_count);
  let idx = 0;
  for (let i = 0; i < colorFeat.length; i++) raw[idx++] = colorFeat[i];
  raw[idx++] = edgeDensity;
  for (let i = 0; i < gradHist.length; i++) raw[idx++] = gradHist[i];
  for (let i = 0; i < hogFeat.length; i++) raw[idx++] = hogFeat[i];

  const scaled = new Float64Array(cfg.var_count);
  for (let i = 0; i < cfg.var_count; i++) {
    scaled[i] = (raw[i] - cfg.feat_mean[i]) / cfg.feat_std[i];
  }

  // RBF kernel value from the input to every support vector.
  const K = new Float64Array(cfg.sv_total);
  const varCount = cfg.var_count;
  for (let s = 0; s < cfg.sv_total; s++) {
    let dist = 0;
    const base = s * varCount;
    for (let d = 0; d < varCount; d++) {
      const diff = scaled[d] - supportVectors[base + d];
      dist += diff * diff;
    }
    K[s] = Math.exp(-cfg.gamma * dist);
  }

  const numClasses = cfg.classes.length;
  const votes = new Array(numClasses).fill(0);
  for (const df of cfg.decision_functions) {
    let sum = -df.rho;
    for (let m = 0; m < df.svidx.length; m++) sum += df.alpha[m] * K[df.svidx[m]];
    if (sum > 0) votes[df.i] += 1;
    else votes[df.j] += 1;
  }

  const totalPairs = cfg.decision_functions.length;
  const probs = votes.map((v) => v / totalPairs);
  return { classes: cfg.classes, probs };
}

// ── UI wiring ─────────────────────────────────────────────────────────────

// The six Intel Scene categories, in a fixed display order (matches the
// notebook's alphabetically-sorted CLASSES list). Used to draw one bar per
// category immediately on page load, before any image is classified.
const SCENE_CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"];

// Draws one <li> bar per category, in a fixed order. Pass probs=null to
// render empty placeholder bars (e.g. before the first classification).
function renderBars(classes, probs, highlightIdx = -1) {
  resultBars.innerHTML = "";
  classes.forEach((cls, i) => {
    const li = document.createElement("li");
    if (i === highlightIdx) li.classList.add("top");
    const pct = probs ? probs[i] * 100 : 0;
    const pctText = probs ? `${pct.toFixed(1)}%` : "—";
    li.innerHTML = `
      <span>${cls}</span>
      <span class="bar-track"><span class="bar-fill" style="width:${pct}%"></span></span>
      <span>${pctText}</span>
    `;
    resultBars.appendChild(li);
  });
}

// Show all six category bars right away, at 0%, so the layout is visible
// before the user classifies anything.
renderBars(SCENE_CLASSES, null);

classifyBtn.addEventListener("click", async () => {
  if (!currentImage) return;
  const model = document.querySelector('input[name="model"]:checked').value;

  classifyBtn.disabled = true;
  setStatus(model === "cnn" ? "Running CNN inference…" : "Extracting classical features and running SVM…");

  const t0 = performance.now();
  try {
    const { classes, probs } = model === "cnn" ? await runCnn(currentImage) : await runSvm(currentImage);
    const elapsed = performance.now() - t0;
    renderResult(classes, probs, model, elapsed);
    setStatus("");
  } catch (err) {
    console.error(err);
    setStatus(
      "Something went wrong: " + err.message +
        " (if you opened this file directly, model files must be served over http:// — see README).",
      true
    );
  } finally {
    classifyBtn.disabled = false;
  }
});

function renderResult(classes, probs, model, elapsedMs) {
  let bestIdx = 0;
  for (let i = 1; i < probs.length; i++) if (probs[i] > probs[bestIdx]) bestIdx = i;

  resultLabel.textContent = classes[bestIdx];
  resultConfidence.textContent =
    model === "cnn"
      ? `${(probs[bestIdx] * 100).toFixed(1)}% confidence`
      : `${Math.round(probs[bestIdx] * 15)}/${15} pairwise votes`;

  // Fixed category order (not sorted by score) so each category always
  // occupies the same row and is easy to compare across classifications.
  renderBars(classes, probs, bestIdx);

  timingEl.textContent = `${model.toUpperCase()} inference took ${elapsedMs.toFixed(0)} ms (client-side).`;
}
