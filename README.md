# Spectral Representation Analysis under Audio Mixture

## 🎯 Objective

This project investigates how different spectral representations behave under signal mixture. Specifically, we compare:

- The transform of a mixture: T(x + y)
- The sum of individual transforms: T(x) + T(y)

The goal is to analyze the limitations of linearity in spectral domains and understand how these affect source separation and invertibility.

---

## 🧠 Motivation

In the time domain, audio signals combine linearly:

x + y

However, in spectral representations (e.g., STFT, mel spectrogram, NSGT), this linearity does not necessarily hold.

Understanding these discrepancies is important for:

- Source separation methods
- Timbre representation
- Inverse reconstruction of audio

---

## 🔬 Research Questions

1. How different are T(x + y) and T(x) + T(y)?
2. Do some representations preserve linearity better than others?
3. How do these differences impact the possibility of separation or inversion?

---

## 🧪 Methodology

### Step 1 — Data
- Select two audio signals: x and y
- Generate mixture: x + y

### Step 2 — Representations
Evaluate multiple spectral transforms:
- STFT
- Mel Spectrogram
- NSGT (if feasible)

### Step 3 — Comparison

For each transform T:

- Compute:
  - T(x)
  - T(y)
  - T(x + y)
  - T(x) + T(y)

- Compare:
  - L2 error
  - Relative difference
  - Visual inspection

---

## 📊 Expected Analysis

- Identify deviations from linearity
- Compare behavior across representations
- Discuss implications for:
  - Source separation
  - Timbre modeling
  - Invertibility

---

## ⚙️ Implementation Plan

### Phase 1
- Implement STFT pipeline
- Compute and visualize comparisons

### Phase 2
- Add mel spectrogram
- Extend analysis

### Phase 3
- Integrate NSGT (optional)
- Compare across all representations

---

## 📅 Timeline

- Early April: Initial experiments (STFT, mel)
- Mid April: NSGT + analysis
- Late April: Writing and consolidation

---

## 🚧 Current Status

- Project initialized
- Problem and methodology defined
- Implementation starting

---

## 📌 Notes

This project focuses on analysis rather than building full separation models. The goal is to understand structural properties of spectral representations.