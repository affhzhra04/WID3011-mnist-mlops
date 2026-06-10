# WID3011 MNIST MLOps Pipeline

[![CI — MNIST MLOps](https://github.com/affhzhra04/WID3011-mnist-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/affhzhra04/WID3011-mnist-mlops/actions/runs)

## Lab Overview
Production-grade deployment pipeline turning an MNIST digit classifier into a containerized microservice serving predictions via a FastAPI endpoint.

### ✅ Lab Completion Checklist

| Part | Task | Done |
| :--- | :--- | :---: |
| **Part 1** | `src/model.py` — CNN architecture | [x] |
| **Part 1** | `src/train.py` — reads from `config.yaml` | [x] |
| **Part 1** | `src/predict.py` — inference utilities | [x] |
| **Part 1** | `configs/config.yaml` — all hyperparameters | [x] |
| **Part 1** | `tests/test_model.py` — output shape + no-NaN tests | [x] |
| **Part 1** | All pytest tests pass | [x] |
| **Part 2** | `src/export_onnx.py` — ONNX export + validation | [x] |
| **Part 2** | ONNX outputs match PyTorch within 1e-4 | [x] |
| **Part 3** | `src/app.py` — FastAPI `/predict` endpoint | [x] |
| **Part 3** | `tests/test_api.py` — status 200 + valid probability | [x] |
| **Part 4** | Dockerfile written and builds successfully | [x] |
| **Part 4** | Container runs and `/health` returns 200 | [x] |
| **Part 4** | curl test against `/predict` succeeds | [x] |
| **Deliverable** | GitHub repo with all files pushed | [x] |
| **Deliverable** | CI passes (green badge on README) | [x] |
