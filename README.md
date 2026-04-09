# TinyML Workshop

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-orange)
![License](https://img.shields.io/badge/License-Apache%202.0-green)

A hands-on introduction to neural network optimization for embedded and
resource-constrained devices, using a simple CNN trained on MNIST.

Topics covered: full-precision training, post-training quantization (PTQ),
quantization-aware training (QAT), magnitude-based pruning, and ONNX export.

---

## What's included

| File | Description |
|---|---|
| `model.py` | CNN architecture |
| `train.py` | Full-precision training |
| `ptq.py` | Post-Training Quantization |
| `qat.py` | Quantization-Aware Training |
| `pruning.py` | Unstructured (magnitude-based) pruning |
| `structured_pruning.py` | Structured (filter) pruning |
| `export_onnx.py` | Export to ONNX format |
| `inference.py` | Compare all model variants |

---

## Requirements

- Python 3.10 or newer — download from https://www.python.org/downloads/
- No GPU required — everything runs on CPU

---

## Getting started

Clone the repository:

```
git clone https://github.com/ArtuurAstaes/tinyml-workshop.git
cd tinyml-workshop
```

Then run the setup script for your OS (do this once):

**Linux / macOS:**
```
bash setup.sh
```

**Windows:**
```
.\setup.bat
```

> **Note for Windows users:** If you get a permission error, Windows may have
> blocked the file. Right-click `setup.bat`, select Properties, and check
> **Unblock** at the bottom, then click OK and try again.

This creates a virtual environment and installs all dependencies. When it
finishes, activate the environment in your own terminal:

**Linux / macOS:**
```
source venv/bin/activate
```

**Windows:**
```
venv\Scripts\activate
```

---

## Running the workshop

Run the scripts in this order. Each one builds on the previous.

```
python train.py                # Train the baseline model (~1 min)
python ptq.py                  # Quantize without retraining (seconds)
python qat.py                  # Quantize with fine-tuning (~1 min)
python unstructured_pruning.py # Unstructured prune + fine-tune (~1 min)
python structured_pruning.py   # Structured prune + fine-tune (~1 min)
python export_onnx.py          # Export to ONNX
python inference.py            # Compare all variants side by side
```

The MNIST dataset (~11 MB) is downloaded automatically on first run.

---

## Expected results

```
Model                     Accuracy    Time (ms)    Size (KB)
------------------------------------------------------------
Baseline (float32)          ~98.1%        ~300ms      215 KB
PTQ (int8)                  ~98.2%        ~275ms       59 KB
QAT (int8)                  ~98.5%        ~275ms       59 KB
Unstructured pruned         ~97.5%        ~300ms      215 KB
Structured pruned           ~97.0%        ~150ms       60 KB
```

Key takeaways:
- **Quantization** reduces model size by ~3.6× with no accuracy loss
- **QAT** slightly outperforms **PTQ** — the model learned to handle quantization noise
- **Unstructured pruning** alone does not reduce file size or speed without a sparse-aware runtime
- **Structured pruning** physically removes filters, giving real size and speed reductions on any hardware

---

## Inspecting the ONNX model

After running `export_onnx.py`, open `cnn.onnx` in **Netron**:
https://netron.app

Drag and drop the file into the browser to visualize the full computation graph,
including layer attributes like `kernel_shape`, `strides`, and `pads`.

---

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
