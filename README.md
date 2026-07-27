# Medicine Strip Classification

A deep learning–powered web application that classifies medicine tablet images using a fine-tuned ResNet-18 model and returns relevant medicine information.

## Project Overview

This project classifies medicine strip / tablet images into one of **76 medicine classes** using transfer learning on ResNet-18 (ImageNet-pretrained). A FastAPI backend serves predictions to a lightweight web frontend that also displays medicine metadata.

The original dataset was collected via the **Medicine Tablet Dataset** on Kaggle and consists of iPhone HEIC photographs of common medicine strips.

## Dataset

The dataset is fetched from Kaggle and currently lives at `dataset/photos/` during development. Use the snippet below to download it with `kagglehub`:

```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("amirbr/medicine-tablet-dataset")

print("Path to dataset files:", path)
```

The downloaded dataset contains `.HEIC` images captured from iPhone photos, organized numerically.

## Model

| Detail               | Value                              |
|----------------------|------------------------------------|
| Architecture         | ResNet-18 (frozen backbone)        |
| Head                 | 1 fully-connected layer            |
| Loss                 | CrossEntropyLoss (label smoothing) |
| Optimizer            | Adam                               |
| Learning Rate        | 1e-3                               |
| Batch Size           | 8                                  |
| Epochs               | 25                                 |
| Classes              | 76 medicines                       |
| Framework            | PyTorch                            |

Trained weights are saved in `model.pth`. Class labels are exported in `classes.json`.

## Repository Structure

```
Medicine-strip-classification/
├── app.py                 # FastAPI backend
├── classes.json           # 76-class label list
├── dataset.py             # PyTorch Dataset loader (HEIC supported)
├── index.html             # Single-page web frontend
├── model.pth              # Trained ResNet-18 weights
├── predict.py             # Standalone inference script
├── train.py               # Model training script
└── requirements.txt       # Python dependencies
```

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/amirbr/Medicine-strip-classification.git
   cd Medicine-strip-classification
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux / macOS
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the FastAPI server:
   ```bash
   uvicorn app:app --reload
   ```

5. Open the application at `http://127.0.0.1:8000`.

## Usage

### Web Interface

- Upload one or more medicine images via drag-and-drop.
- View the top-3 predicted classes, confidence scores, and medicine metadata (composition, usage, side effects).

### Standalone Prediction

```bash
python predict.py path/to/image.jpg
```

### Retrain the Model

```bash
python train.py
```

The training script looks for images in the `images/` directory (or `dataset/photos/` depending on configuration), trains for 25 epochs, and saves `model.pth` and `classes.json`.

## API Endpoints

| Method | Endpoint               | Description                               |
|--------|------------------------|-------------------------------------------|
| GET    | `/`                    | Serves the frontend                       |
| GET    | `/api/medicines`       | Returns sorted list of medicine names     |
| POST   | `/predict`             | Predicts labels for uploaded images       |
| POST   | `/interactions-from-images` | Placeholder (interaction model disabled) |

## Dependencies

- `torch`
- `torchvision`
- `pillow`
- `pillow-heif`
- `tqdm`
- `fastapi`
- `uvicorn`
- `python-multipart`

## Notes

- Training images are HEIC (iPhone) format. `pillow-heif` is required for loading them.
- The backbone layers are frozen during training; only the classification head is updated.
- `dataset/medicine_info.csv` is expected by `app.py` for medicine metadata but is not included in this repository.
- The `/interactions-from-images` endpoint is a stub and returns a disabled message.
