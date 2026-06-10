# WID3011 MNIST MLOps Pipeline

[![CI — MNIST MLOps](https://github.com/affhzhra04/WID3011-mnist-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/affhzhra04/WID3011-mnist-mlops/actions/runs)

## Lab Overview
Production-grade deployment pipeline turning an MNIST digit classifier into a containerized microservice serving predictions via a FastAPI endpoint.

### ✅ Lab Completion Checklist

| Part | Task | Done |
| :--- | :--- | :---: |
| **Part 1** | `src/model.py` — CNN architecture | [/] |
| **Part 1** | `src/train.py` — reads from `config.yaml` | [/] |
| **Part 1** | `src/predict.py` — inference utilities | [/] |
| **Part 1** | `configs/config.yaml` — all hyperparameters | [/] |
| **Part 1** | `tests/test_model.py` — output shape + no-NaN tests | [/] |
| **Part 1** | All pytest tests pass | [/] |
| **Part 2** | `src/export_onnx.py` — ONNX export + validation | [/] |
| **Part 2** | ONNX outputs match PyTorch within 1e-4 | [/] |
| **Part 3** | `src/app.py` — FastAPI `/predict` endpoint | [/] |
| **Part 3** | `tests/test_api.py` — status 200 + valid probability | [/] |
| **Part 4** | Dockerfile written and builds successfully | [/] |
| **Part 4** | Container runs and `/health` returns 200 | [/] |
| **Part 4** | curl test against `/predict` succeeds | [/] |
| **Deliverable** | GitHub repo with all files pushed | [/] |
| **Deliverable** | CI passes (green badge on README) | [/] |
