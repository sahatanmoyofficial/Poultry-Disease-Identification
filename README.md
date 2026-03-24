# 🐔 Chicken Disease Classification Project (CDCP) — Poultry Disease Identification

> **Binary image classification of poultry disease — VGG16 transfer learning (Coccidiosis vs Healthy) with a production-grade 4-stage MLOps pipeline: DVC-tracked data ingestion → base model preparation → training → evaluation, served via Flask REST API, containerised with Docker, and deployed to both AWS (ECR + EC2 self-hosted runner) and Azure (ACR + Azure Web App) via dual GitHub Actions CI/CD workflows**
>
> A complete end-to-end MLOps system: Chicken fecal image dataset → `config.yaml` / `params.yaml` → entity dataclasses → `ConfigurationManager` → components → pipeline stages → DVC DAG → Flask `/predict` (base64 image → Coccidiosis/Healthy) → Dockerfile → GitHub Actions → AWS ECR/EC2 + Azure Web App.

---

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange?logo=tensorflow)](https://tensorflow.org/)
[![VGG16](https://img.shields.io/badge/Model-VGG16-brightgreen)](https://keras.io/api/applications/vgg/)
[![Flask](https://img.shields.io/badge/API-Flask-lightgrey?logo=flask)](https://flask.palletsprojects.com/)
[![DVC](https://img.shields.io/badge/MLOps-DVC-purple)](https://dvc.org/)
[![Docker](https://img.shields.io/badge/Container-Docker-blue?logo=docker)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

## 📊 Project Slides

👉 **[View the Project Presentation (PPTX)](https://docs.google.com/presentation/d/1IJZSBoK4DbdRbq4yc-PeOw_ioM6nkrJT/edit?usp=sharing&ouid=117459468470211543781&rtpof=true&sd=true)**

---

## 📋 Table of Contents

| # | Section |
|---|---------|
| 1 | [Problem Statement](#1-problem-statement) |
| 2 | [Project Overview](#2-project-overview) |
| 3 | [Tech Stack](#3-tech-stack) |
| 4 | [System Architecture](#4-system-architecture) |
| 5 | [Repository Structure](#5-repository-structure) |
| 6 | [Dataset](#6-dataset) |
| 7 | [VGG16 Transfer Learning](#7-vgg16-transfer-learning) |
| 8 | [4-Stage MLOps Pipeline](#8-4-stage-mlops-pipeline) |
| 9 | [DVC Pipeline & DAG](#9-dvc-pipeline--dag) |
| 10 | [Config Management Pattern](#10-config-management-pattern) |
| 11 | [Training & Evaluation Results](#11-training--evaluation-results) |
| 12 | [Flask REST API](#12-flask-rest-api) |
| 13 | [Docker Containerisation](#13-docker-containerisation) |
| 14 | [Dual CI/CD — AWS + Azure](#14-dual-cicd--aws--azure) |
| 15 | [How to Replicate](#15-how-to-replicate) |
| 16 | [Business Applications](#16-business-applications) |
| 17 | [How to Improve This Project](#17-how-to-improve-this-project) |
| 18 | [Troubleshooting](#18-troubleshooting) |
| 19 | [Glossary](#19-glossary) |

---

## 1. Problem Statement

### What problem are we solving?

Coccidiosis is a highly contagious parasitic disease affecting the intestinal tract of chickens, caused by *Eimeria* species. In commercial poultry farming:

- **Economic impact:** Coccidiosis costs the global poultry industry over **$3 billion annually** in mortality, reduced weight gain, and medication costs
- **Detection challenge:** Early infection is invisible to the naked eye; current diagnostics require fecal microscopy or PCR — expensive, time-consuming, and requiring specialist equipment
- **Scale:** A single poultry house may contain 20,000–50,000 birds; daily individual inspection is operationally impossible

Automated image-based screening of chicken fecal samples can detect coccidiosis early — before clinical signs appear — enabling targeted treatment and reducing whole-flock medication.

### What does CDCP classify?

> *"Given a chicken fecal image — is this Coccidiosis (infected) or Healthy?"*

| Class | Index | Description |
|-------|-------|-------------|
| **Coccidiosis** | 0 | Fecal sample shows coccidial oocysts indicative of Eimeria infection |
| **Healthy** | 1 | Normal fecal sample — no signs of parasitic infection |

---

## 2. Project Overview

| Aspect | Detail |
|--------|--------|
| **Task** | Binary image classification: Coccidiosis vs Healthy |
| **Base model** | VGG16 (frozen) + custom Dense head |
| **Input shape** | 224 × 224 × 3 (as per VGG16 specification) |
| **Training epochs** | 1 (demo run) |
| **Batch size** | 16 |
| **Validation split (training)** | 20% |
| **Validation split (evaluation)** | 30% |
| **Optimiser** | SGD (learning_rate=0.01) |
| **Loss** | Categorical Crossentropy |
| **Augmentation** | rotation=40, h_flip, width/height shift=0.2, shear=0.2, zoom=0.2 |
| **Evaluation scores** | Loss: 1.8001 · Accuracy: 73.28% |
| **MLOps** | DVC 4-stage pipeline: DataIngestion → PrepareBaseModel → Training → Evaluation |
| **API** | Flask: GET `/` · GET/POST `/train` · POST `/predict` |
| **Serving format** | Base64 image in → JSON `{"image": "Coccidiosis"/"Healthy"}` out |
| **Container** | Docker (python:3.8-slim-buster + AWS CLI) |
| **CI/CD** | Dual: AWS ECR → EC2 self-hosted runner + Azure ACR → Azure Web App |

---

## 3. Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Language** | Python 3.8 | Core language |
| **Deep Learning** | TensorFlow / Keras | VGG16 model, training, evaluation |
| **Base model** | VGG16 (ImageNet weights) | Frozen transfer learning backbone |
| **MLOps pipeline** | DVC 0.x | Stage orchestration, dependency tracking, artifact caching |
| **Config management** | `python-box` (`ConfigBox`) + `PyYAML` | Dot-notation YAML access |
| **Type enforcement** | `ensure` (`@ensure_annotations`) | Runtime type checking on utils |
| **Web framework** | Flask + Flask-CORS | REST API — `/train`, `/predict` |
| **Serialisation** | Base64 | Image transfer: client encodes → server decodes → saves as `inputImage.jpg` |
| **Data pipeline** | `urllib.request` + `zipfile` | Download + extract fecal image dataset |
| **Training callbacks** | TensorBoard + ModelCheckpoint | Logging + `save_best_only=True` weight saving |
| **Containerisation** | Docker (`python:3.8-slim-buster`) | Reproducible deployment environment |
| **Package management** | `setup.py` (`src/` layout) | Installs `cnnClassifier` as importable package |
| **Logging** | Python `logging` | Per-module named loggers |
| **CI/CD (AWS)** | GitHub Actions → ECR → EC2 self-hosted | Build image → push to ECR → pull on EC2 → run container |
| **CI/CD (Azure)** | GitHub Actions → ACR → Azure Web App | Build → push to ACR → deploy to Web App |

---

## 4. System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                    CONFIG LAYER                                      │
│                                                                      │
│  config/config.yaml          params.yaml                            │
│         │                         │                                  │
│  read_yaml() → ConfigBox    read_yaml() → ConfigBox                 │
│         └─────────────────────────┘                                 │
│                       │                                              │
│         ConfigurationManager                                         │
│   ├── get_data_ingestion_config()   → DataIngestionConfig           │
│   ├── get_prepare_base_model_config() → PrepareBaseModelConfig      │
│   ├── get_prepare_callback_config() → PrepareCallbacksConfig        │
│   ├── get_training_config()         → TrainingConfig                │
│   └── get_validation_config()       → EvaluationConfig              │
└─────────────────────────────────────────────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────────────┐
│                    DVC 4-STAGE PIPELINE (dvc.yaml)                   │
│                                                                      │
│  Stage 1: data_ingestion                                             │
│    urllib.request.urlretrieve() → data.zip → extractall()           │
│    → artifacts/data_ingestion/Chicken-fecal-images/                 │
│                                                                      │
│  Stage 2: prepare_base_model                                         │
│    VGG16(imagenet, include_top=False, 224×224×3)                    │
│    → Flatten → Dense(2, softmax) → SGD(lr=0.01)                    │
│    → base_model.h5 + base_model_updated.h5                          │
│                                                                      │
│  Stage 3: training                                                   │
│    load base_model_updated.h5 → ImageDataGenerator (augmented)     │
│    → callbacks: TensorBoard + ModelCheckpoint(save_best_only=True)  │
│    → model.fit(1 epoch, batch=16, val_split=0.20)                  │
│    → artifacts/training/model.h5                                    │
│                                                                      │
│  Stage 4: evaluation                                                 │
│    load model.h5 → ImageDataGenerator(val_split=0.30)              │
│    → model.evaluate() → scores.json {loss, accuracy}               │
│                                                                      │
│  Results: loss=1.8001, accuracy=73.28%                              │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────────────┐
│                    FLASK REST API (app.py)                           │
│                                                                      │
│  GET  /              → render templates/index.html                  │
│  GET/POST /train     → os.system("python main.py") → "Training done"│
│  POST /predict       → decodeImage(base64) → inputImage.jpg         │
│                        → PredictionPipeline.predict()               │
│                        → [{"image": "Coccidiosis"/"Healthy"}]       │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────────────┐
│             DOCKER + DUAL CI/CD                                      │
│                                                                      │
│  Dockerfile: python:3.8-slim-buster + awscli + pip install          │
│                                                                      │
│  AWS (main.yaml):                                                    │
│    push to main → lint/test → ECR build+push                        │
│    → EC2 self-hosted runner → docker pull + run (-p 8080:8080)      │
│                                                                      │
│  Azure (main_chickenapp.yml):                                        │
│    push to main → ACR build+push → Azure Web App deploy             │
│    app runs on port 80 (app.py: app.run(port=80))                   │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 5. Repository Structure

```
Chicken-Disease-Classification/
│
├── src/cnnClassifier/             # Core package (src/ layout)
│   ├── constants/__init__.py      # CONFIG_FILE_PATH · PARAMS_FILE_PATH
│   ├── entity/config_entity.py    # @dataclass configs (frozen=True) per stage
│   ├── config/configuration.py   # ConfigurationManager
│   ├── utils/common.py            # read_yaml · create_directories · decodeImage · encodeImageIntoBase64
│   ├── components/
│   │   ├── data_ingestion.py      # DataIngestion: download_file + extract_zip_file
│   │   ├── prepare_base_model.py  # PrepareBaseModel: VGG16 + custom head
│   │   ├── prepare_callbacks.py   # PrepareCallback: TensorBoard + ModelCheckpoint
│   │   ├── training.py            # Training: ImageDataGenerator + model.fit
│   │   └── evaluation.py          # Evaluation: model.evaluate → scores.json
│   └── pipeline/
│       ├── stage_01_data_ingestion.py
│       ├── stage_02_prepare_base_model.py
│       ├── stage_03_training.py
│       ├── stage_04_evaluation.py
│       └── predict.py             # PredictionPipeline: load model.h5 → argmax
│
├── config/config.yaml             # Artifact paths + source URL
├── params.yaml                    # All tunable training hyperparameters
├── dvc.yaml                       # DVC stage definitions + deps/outs/params/metrics
├── dvc.lock                       # Locked pipeline state (reproducibility)
├── scores.json                    # Evaluation output: loss + accuracy
├── app.py                         # Flask API (3 routes)
├── main.py                        # Full pipeline runner (all 4 stages)
├── Dockerfile                     # python:3.8-slim-buster + awscli
├── setup.py                       # Package: cnnClassifier (src/ layout)
├── requirements.txt               # All dependencies
├── templates/index.html           # Flask frontend HTML
├── architecture.jpg               # VGG16 architecture diagram
├── inputImage.jpg                 # Temporary inference image
├── metadata.txt                   # Project learning outcomes
│
├── .dvc/config                    # DVC remote configuration
├── .github/workflows/
│   ├── main.yaml                  # AWS CI/CD: ECR → EC2 self-hosted runner
│   └── main_chickenapp.yml        # Azure CI/CD: ACR → Azure Web App
│
└── research/
    ├── 01_data_ingestion.ipynb
    ├── 02_prepare_base_model.ipynb
    ├── 03_prepare_callbacks.ipynb
    ├── 04_training.ipynb
    ├── 05_model_evaluation.ipynb
    └── trials.ipynb
```

---

## 6. Dataset

| Property | Detail |
|----------|--------|
| **Name** | Chicken Fecal Images (Chicken-fecal-images) |
| **Source URL** | `https://github.com/entbappy/Branching-tutorial/raw/master/Chicken-fecal-images.zip` |
| **Format** | `.jpg` images organised by class subdirectory |
| **Classes** | `Coccidiosis/` · `Healthy/` |
| **Task** | Binary classification |
| **Download** | Automated via `DataIngestion.download_file()` at pipeline Stage 1 |
| **Extraction** | `zipfile.ZipFile.extractall()` → `artifacts/data_ingestion/Chicken-fecal-images/` |

The dataset is downloaded automatically during Stage 1 — no manual download required. If the zip file already exists locally, Stage 1 logs its size and skips re-download.

---

## 7. VGG16 Transfer Learning

### Architecture

```
Input (224×224×3)
         │
VGG16 backbone (frozen, weights='imagenet', include_top=False)
         │  ← All 13 conv layers + 5 max-pool layers frozen
         │  ← No top Dense/softmax layers (include_top=False)
         │
Flatten()  ← Collapse spatial dimensions to 1D vector
         │
Dense(2, activation='softmax')  ← Binary output: [Coccidiosis, Healthy]
         │
Compiled with:
  optimizer = SGD(learning_rate=0.01)
  loss      = CategoricalCrossentropy()
  metrics   = ['accuracy']
```

### Why VGG16?

VGG16 was chosen as the backbone because:
- **Proven ImageNet features** — its 13 conv layers encode rich hierarchical features (edges → textures → shapes → objects) from 14M+ ImageNet images
- **Simple architecture** — uniform 3×3 convolutions make it easier to understand and debug than more complex architectures
- **Medical/microscopy transfer** — ImageNet texture features transfer well to biological image classification tasks including microscopy and fecal imaging
- **Frozen training** — with `freeze_all=True`, only the 2-unit Dense head (~32K params) trains; the entire VGG16 backbone (~138M params) stays fixed

### Model Saving

Two model files are saved during Stage 2:

| File | Content |
|------|---------|
| `artifacts/prepare_base_model/base_model.h5` | Raw VGG16 (no top, frozen, no custom head) |
| `artifacts/prepare_base_model/base_model_updated.h5` | VGG16 + Flatten + Dense(2) head — ready for training |

Stage 3 loads `base_model_updated.h5` and fine-tunes only the head.

---

## 8. 4-Stage MLOps Pipeline

### Stage 1 — Data Ingestion

**Component:** `data_ingestion.py`

```python
def download_file(self):
    if not os.path.exists(self.config.local_data_file):
        filename, headers = request.urlretrieve(
            url=self.config.source_URL,
            filename=self.config.local_data_file
        )

def extract_zip_file(self):
    with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
        zip_ref.extractall(self.config.unzip_dir)
```

Downloads `Chicken-fecal-images.zip` and extracts to `artifacts/data_ingestion/Chicken-fecal-images/`.

---

### Stage 2 — Prepare Base Model

**Component:** `prepare_base_model.py`

```python
def get_base_model(self):
    self.model = tf.keras.applications.vgg16.VGG16(
        input_shape=(224, 224, 3),
        weights='imagenet',
        include_top=False
    )
    self.save_model(path=self.config.base_model_path, model=self.model)

def update_base_model(self):
    self.full_model = self._prepare_full_model(
        model=self.model,
        classes=2,
        freeze_all=True,     # All VGG16 layers frozen
        freeze_till=None,
        learning_rate=0.01
    )
    self.save_model(path=self.config.updated_base_model_path, model=self.full_model)
```

`_prepare_full_model` adds `Flatten() → Dense(2, softmax)` on top of the frozen VGG16 and compiles with SGD.

---

### Stage 3 — Training

**Component:** `training.py`

```python
# Training data augmentation (AUGMENTATION=True)
train_datagenerator = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.20,
    rotation_range=40,
    horizontal_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2
)

# Callbacks
callbacks = [
    TensorBoard(log_dir=f"tb_logs_at_{timestamp}"),
    ModelCheckpoint(filepath=checkpoint_path, save_best_only=True)
]
```

Trains for 1 epoch (demo). `ModelCheckpoint(save_best_only=True)` saves only when validation accuracy improves — important for multi-epoch runs.

---

### Stage 4 — Evaluation

**Component:** `evaluation.py`

```python
# Note: evaluation uses validation_split=0.30 (different from training's 0.20)
datagenerator_kwargs = dict(
    rescale=1./255,
    validation_split=0.30
)
```

Evaluates on a 30% hold-out split and saves results to `scores.json`.

---

## 9. DVC Pipeline & DAG

### `dvc.yaml` — Stage Definitions

```yaml
stages:
  data_ingestion:
    cmd: python src/cnnClassifier/pipeline/stage_01_data_ingestion.py
    deps: [stage_01_data_ingestion.py, config/config.yaml]
    outs: [artifacts/data_ingestion/Chicken-fecal-images]

  prepare_base_model:
    cmd: python src/cnnClassifier/pipeline/stage_02_prepare_base_model.py
    deps: [stage_02_prepare_base_model.py, config/config.yaml]
    params: [IMAGE_SIZE, INCLUDE_TOP, CLASSES, WEIGHTS, LEARNING_RATE]
    outs: [artifacts/prepare_base_model]

  training:
    cmd: python src/cnnClassifier/pipeline/stage_03_training.py
    deps: [..., artifacts/data_ingestion/Chicken-fecal-images, artifacts/prepare_base_model]
    params: [IMAGE_SIZE, EPOCHS, BATCH_SIZE, AUGMENTATION]
    outs: [artifacts/training/model.h5]

  evaluation:
    cmd: python src/cnnClassifier/pipeline/stage_04_evaluation.py
    deps: [..., artifacts/training/model.h5]
    params: [IMAGE_SIZE, BATCH_SIZE]
    metrics:
      - scores.json: {cache: false}
```

### DVC DAG

```
data_ingestion
      │
      ▼
prepare_base_model
      │
      ▼
training  ──────────────────────►  evaluation
      │                                │
      ▼                                ▼
artifacts/training/model.h5     scores.json
```

### DVC Commands

```bash
dvc init          # Initialise DVC in project
dvc repro         # Re-run only stages whose deps/params have changed
dvc dag           # Visualise the pipeline graph
```

**Key benefit:** DVC caches intermediate artifacts (`Chicken-fecal-images/`, `prepare_base_model/`, `model.h5`). Only changed stages re-run. If `params.yaml` changes EPOCHS from 1 to 10, only `training` and `evaluation` re-run — `data_ingestion` and `prepare_base_model` use cache.

---

## 10. Config Management Pattern

This project implements the same clean 5-layer MLOps config pattern as the NLPSum project:

### Layer 1 — YAML Sources

```yaml
# config.yaml — artifact paths and URLs
training:
  root_dir: artifacts/training
  trained_model_path: artifacts/training/model.h5

# params.yaml — all tunable hyperparameters
AUGMENTATION: True
IMAGE_SIZE: [224, 224, 3]
BATCH_SIZE: 16
EPOCHS: 1
CLASSES: 2
WEIGHTS: imagenet
LEARNING_RATE: 0.01
```

### Layer 2 — Constants

```python
CONFIG_FILE_PATH = Path("config/config.yaml")
PARAMS_FILE_PATH = Path("params.yaml")
```

### Layer 3 — Entity Dataclasses

```python
@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    training_data: Path
    params_epochs: int
    params_batch_size: int
    params_is_augmentation: bool
    params_image_size: list
```

All config entity classes use `@dataclass(frozen=True)` — making them immutable once created, preventing accidental modification during pipeline execution.

### Layer 4 — ConfigurationManager

```python
class ConfigurationManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)   # → ConfigBox (dot-notation)
        self.params = read_yaml(params_filepath)   # → ConfigBox
        create_directories([self.config.artifacts_root])
```

### Layer 5 — Components consume typed configs

```python
class Training:
    def __init__(self, config: TrainingConfig):  # ← typed frozen dataclass
        self.config = config
```

---

## 11. Training & Evaluation Results

### Current Results (1-epoch demo run)

| Metric | Value |
|--------|-------|
| **Loss** | 1.8001 |
| **Accuracy** | **73.28%** |
| **Epochs trained** | 1 |
| **Model** | VGG16 (frozen) + Dense(2) head |
| **Optimiser** | SGD (lr=0.01) |

### Why 73.28% Accuracy?

The current results reflect a minimal 1-epoch demonstration run:

1. **Only 1 epoch** — the Dense head has had minimal gradient updates; SGD with lr=0.01 needs multiple epochs to converge
2. **SGD vs Adam** — SGD is a valid choice for fine-tuning frozen models but converges more slowly than Adam; requires more epochs and careful LR scheduling
3. **High loss (1.8001)** — suggests the model is still far from convergence; typical val loss for a converged binary classifier should be < 0.5
4. **Expected performance** — a fully trained VGG16 on this binary task with 10–20 epochs should reach 85–95% accuracy

### `scores.json`

```json
{
    "loss": 1.8000835180282593,
    "accuracy": 0.732758641242981
}
```

This file is tracked as a DVC metric (`cache: false`), enabling `dvc metrics show` to display performance across pipeline runs.

---

## 12. Flask REST API

### Routes

| Method | Route | Description |
|--------|-------|-------------|
| `GET` | `/` | Renders `templates/index.html` — web frontend |
| `GET/POST` | `/train` | Triggers `os.system("python main.py")` — runs full 4-stage pipeline |
| `POST` | `/predict` | Accepts base64 image JSON → returns `[{"image": "Coccidiosis"/"Healthy"}]` |

### Prediction Pipeline (`pipeline/predict.py`)

```python
class PredictionPipeline:
    def predict(self):
        # Load trained model
        model = load_model(os.path.join("artifacts", "training", "model.h5"))

        # Preprocess: load saved inputImage.jpg, resize to 224×224
        test_image = image.load_img(self.filename, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)   # (1, 224, 224, 3)

        # Predict: argmax of softmax output
        result = np.argmax(model.predict(test_image), axis=1)

        # Class mapping: 0=Coccidiosis, 1=Healthy
        if result[0] == 1:
            return [{"image": "Healthy"}]
        else:
            return [{"image": "Coccidiosis"}]
```

### Base64 Image Flow

```
Client (browser/mobile):
  1. Capture/load image
  2. base64-encode → send in JSON body: {"image": "<base64_string>"}

Server (Flask /predict):
  1. decodeImage(base64_string, "inputImage.jpg")  → writes file
  2. PredictionPipeline("inputImage.jpg").predict() → loads file
  3. Returns: [{"image": "Coccidiosis"}]
```

### `decodeImage` / `encodeImageIntoBase64` (utils/common.py)

```python
def decodeImage(imgstring, fileName):
    imgdata = base64.b64decode(imgstring)
    with open(fileName, 'wb') as f:
        f.write(imgdata)

def encodeImageIntoBase64(croppedImagePath):
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())
```

---

## 13. Docker Containerisation

### Dockerfile

```dockerfile
FROM python:3.8-slim-buster

RUN apt update -y && apt install awscli -y
WORKDIR /app

COPY . /app
RUN pip install -r requirements.txt

CMD ["python3", "app.py"]
```

**Notes:**
- `python:3.8-slim-buster` — minimal Debian Buster image with Python 3.8, matching the `conda create -n cnncls python=3.8` setup
- `awscli` installed at build time — required for the EC2 deployment step to interact with AWS services
- `CMD ["python3", "app.py"]` — starts Flask on `host='0.0.0.0', port=80` (Azure configuration)
- Port mapping differs by cloud: EC2 uses `docker run -p 8080:8080`, Azure Web App routes port 80 directly

### Build & Run Locally

```bash
docker build -t cdcp:latest .
docker run -d -p 8080:80 cdcp:latest
# Access at http://localhost:8080
```

---

## 14. Dual CI/CD — AWS + Azure

This project implements **two simultaneous CI/CD pipelines** — a rare and production-grade feature demonstrating multi-cloud deployment capability.

### Pipeline 1 — AWS (`main.yaml`)

```
Trigger: push to main (ignoring README.md changes)

Job 1: integration (ubuntu-latest)
  └── Checkout code
  └── Lint (placeholder: echo)
  └── Run unit tests (placeholder: echo)

Job 2: build-and-push-ecr-image (ubuntu-latest, needs: integration)
  └── Checkout code
  └── Configure AWS credentials (secrets: ACCESS_KEY, SECRET_KEY, REGION)
  └── Login to Amazon ECR
  └── docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:latest .
  └── docker push → ECR

Job 3: Continuous-Deployment (self-hosted EC2 runner, needs: build-and-push-ecr-image)
  └── Checkout code
  └── Configure AWS credentials
  └── Login to Amazon ECR
  └── docker pull ECR_URI:latest
  └── docker run -d -p 8080:8080 --name=cnncls ... ECR_URI:latest
  └── docker system prune -f (clean old images)
```

**Required GitHub Secrets:**

| Secret | Description |
|--------|-------------|
| `AWS_ACCESS_KEY_ID` | IAM user access key |
| `AWS_SECRET_ACCESS_KEY` | IAM user secret key |
| `AWS_REGION` | e.g., `us-east-1` |
| `AWS_ECR_LOGIN_URI` | e.g., `566373416292.dkr.ecr.ap-south-1.amazonaws.com` |
| `ECR_REPOSITORY_NAME` | ECR repository name (e.g., `simple-app`) |

**AWS IAM Policies required:** `AmazonEC2ContainerRegistryFullAccess` + `AmazonEC2FullAccess`

---

### Pipeline 2 — Azure (`main_chickenapp.yml`)

```
Trigger: push to main (or manual workflow_dispatch)

Job 1: build (ubuntu-latest)
  └── Checkout code
  └── Set up Docker Buildx
  └── Login to ACR: chickenapp.azurecr.io
  └── docker build + push → chickenapp.azurecr.io/chicken:{sha}

Job 2: deploy (ubuntu-latest, needs: build)
  └── azure/webapps-deploy@v2
      app-name: chickenapp
      slot-name: production
      images: chickenapp.azurecr.io/chicken:{sha}
```

**Required GitHub Secrets:**

| Secret | Description |
|--------|-------------|
| `AzureAppService_ContainerUsername_*` | ACR username |
| `AzureAppService_ContainerPassword_*` | ACR password (store in Secrets, NEVER in README) |
| `AzureAppService_PublishProfile_*` | Azure Web App publish profile XML |

---

## 15. How to Replicate

### Step 1 — Clone and Install

```bash
git clone https://github.com/sahatanmoyofficial/Chicken-Disease-Classification--Project.git
cd Chicken-Disease-Classification--Project

conda create -n cnncls python=3.8 -y
conda activate cnncls

pip install -r requirements.txt
pip install -e .   # Install cnnClassifier package in editable mode
```

### Step 2 — Run Full Pipeline

```bash
# Option A: Direct Python
python main.py

# Option B: DVC (recommended — uses caching)
dvc init
dvc repro

# Visualise the DAG
dvc dag
```

### Step 3 — Start the Flask API

```bash
python app.py
# Opens at http://0.0.0.0:80 (Azure) or modify port to 8080 for local
```

### Step 4 — Test the Prediction Endpoint

```python
import requests, base64

with open("inputImage.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

response = requests.post(
    "http://localhost:8080/predict",
    json={"image": img_b64}
)
print(response.json())
# [{"image": "Coccidiosis"}] or [{"image": "Healthy"}]
```

---

## 16. Business Applications

| Stakeholder | Application |
|-------------|------------|
| **Poultry farms** | Daily automated screening of fecal samples from litter trays — flag infected flocks for targeted treatment |
| **Veterinary clinics** | Rapid diagnostic screening — reduce turnaround time from days (PCR) to seconds |
| **Feed & pharma companies** | Efficacy monitoring — track coccidiosis prevalence before/after medication |
| **Biosecurity auditors** | Ongoing farm-level disease surveillance without specialist presence |
| **Research institutes** | Large-scale epidemiological studies on *Eimeria* species distribution |

### Generalises to Other Poultry Diseases

The same VGG16 + 4-stage DVC pipeline applies to:

| Disease | Visual Indicator | Modified Classes |
|---------|-----------------|-----------------|
| **Newcastle Disease** | Neurological symptoms on feces | Healthy / Infected |
| **Avian Influenza** | Fecal and respiratory changes | Low-Path / High-Path |
| **Marek's Disease** | Tumour detection in tissue images | Benign / Malignant |
| **Aspergillosis** | Respiratory lesion images | Healthy / Infected |

---

## 17. How to Improve This Project

### 🧠 Model Improvements

| Area | Priority | Recommendation |
|------|----------|---------------|
| **Increase training epochs** | 🔴 High | Change `EPOCHS: 1` to `EPOCHS: 15` in `params.yaml` — current 73.28% will improve significantly; `dvc repro` will only re-run training + evaluation |
| **Switch optimiser to Adam** | 🔴 High | Replace `SGD(lr=0.01)` with `Adam(lr=1e-4)` in `prepare_base_model.py` — Adam converges faster for transfer learning |
| **Add per-class metrics** | 🟡 Medium | Current evaluation reports only overall loss + accuracy — add `classification_report` for precision/recall per class |
| **Fine-tune top VGG16 layers** | 🟡 Medium | After head training, unfreeze top 2–4 VGG16 blocks with lr ~1e-5 for domain adaptation |

### 🏗️ Engineering Improvements

| Area | Recommendation |
|------|---------------|
| **Remove plaintext password from README** | `s3cEZKH5yy...` appears in the original README — rotate the Azure ACR secret and store in GitHub Secrets only |
| **Add real CI tests** | `main.yaml` has `echo "Running unit tests"` placeholder — add `pytest tests/` |
| **Fix `/train` endpoint** | `os.system("python main.py")` blocks the Flask thread during multi-minute training — use `subprocess.Popen` with async response |
| **Align validation splits** | `training.py` uses `validation_split=0.20` but `evaluation.py` uses `validation_split=0.30` — standardise to avoid inconsistent metrics |
| **Update VGG16 call** | `tf.keras.applications.vgg16.VGG16` is legacy — use `tf.keras.applications.VGG16` |
| **Add model versioning** | Save models with timestamps (`model_{timestamp}.h5`) or use DVC model registry |

---

## 18. Troubleshooting

| Error / Symptom | Fix |
|----------------|-----|
| `ModuleNotFoundError: cnnClassifier` | Run `pip install -e .` to install the package in editable mode |
| `FileNotFoundError: artifacts/training/model.h5` | Run `python main.py` or `dvc repro` first to generate the trained model |
| Flask `/predict` returns empty response | `inputImage.jpg` may not be written correctly — verify `decodeImage()` call with valid base64 string |
| Training loss very high after 1 epoch | Expected — change `EPOCHS: 1` to `EPOCHS: 15` in `params.yaml` |
| AWS deployment fails at `docker run` | Check EC2 security group allows inbound TCP on port 8080 |
| Azure deployment 502 Bad Gateway | App running on port 80 — verify Azure Web App port setting matches `app.run(port=80)` |
| DVC not tracking changes | Run `dvc repro` — DVC compares MD5 of deps/params, not timestamps |

---

## 19. Glossary

| Term | Definition |
|------|-----------|
| **Coccidiosis** | A parasitic disease of poultry caused by *Eimeria* species; highly contagious, spreads via fecal-oral route |
| **VGG16** | 16-layer deep CNN (13 conv + 3 fully connected) pre-trained on ImageNet; `include_top=False` removes the FC layers for transfer learning |
| **DVC** | Data Version Control — Git extension for versioning ML data, models, and pipeline stages; `dvc repro` re-runs only changed stages |
| **`dvc.yaml`** | Stage definitions: `cmd`, `deps` (dependencies), `outs` (outputs), `params` (tracked hyperparameters), `metrics` (evaluation outputs) |
| **`dvc.lock`** | Locked state of the pipeline — stores MD5 hashes of all deps/outs; enables exact reproduction of any pipeline run |
| **DVC DAG** | Directed Acyclic Graph — visual representation of stage dependencies; `dvc dag` prints it |
| **`@dataclass(frozen=True)`** | Python immutable dataclass — prevents accidental mutation of config objects after creation |
| **`ConfigBox`** | `python-box` object enabling dot-notation access to nested YAML: `config.training.root_dir` vs `config["training"]["root_dir"]` |
| **`@ensure_annotations`** | Decorator from `ensure` library validating function argument types at runtime |
| **`ModelCheckpoint(save_best_only=True)`** | Keras callback saving model only when validation metric improves across epochs |
| **TensorBoard** | TensorFlow visualisation tool for training metrics — log dir created with timestamp per run |
| **ECR** | Amazon Elastic Container Registry — managed Docker image registry in AWS |
| **Self-hosted runner** | An EC2 instance configured to receive GitHub Actions jobs; pulls Docker image from ECR and runs it |
| **ACR** | Azure Container Registry — managed Docker image registry in Azure |
| **Azure Web App** | PaaS container hosting — pulls image from ACR, runs it with port 80 exposed |
| **Base64 encoding** | Converts binary data (image bytes) to ASCII text safe for JSON transmission; `decodeImage` reverses this |
| **`src/` layout** | Python packaging convention where source code lives under `src/` — prevents accidental import without installation |
| **`flask_cors`** | Flask extension enabling Cross-Origin Resource Sharing — allows browser clients from different domains to call the API |

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Tanmoy Saha**
[linkedin.com/in/sahatanmoyofficial](https://linkedin.com/in/sahatanmoyofficial) | sahatanmoyofficial@gmail.com

---