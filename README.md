# Solar Panel Dust Dectation

Binary image classifier that distinguishes **Clean** from **Dusty** solar panel surfaces. Two architectures were trained and compared — a custom CNN and a MobileNetV3Small transfer learning model — with the best checkpoint deployed as a Streamlit web app.

---

## Results

| Model | Test Accuracy | Params |
|---|---|---|
| Custom CNN | 77.52% | 322,081 |
| **MobileNetV3Small (TL)** | **84.50%** | 1,087,089 |

The transfer learning model is used in the deployed app.

---

## Project Structure

```
.
├── solar-panel-ipynb.ipynb          # Training notebook (data prep, training, evaluation)
├── tl_feature_extraction_best.keras # Best MobileNetV3Small checkpoint — used by the app
├── app.py                           # Streamlit inference app
└── README.md
```

> The `models/` directory (containing `custom_cnn.keras` and `mobilenetv3_transfer.keras`) is generated locally when the notebook is run and is not included in this repository. The app only requires `tl_feature_extraction_best.keras`.

---

## Dataset

**Source:** [Solar Panel Dust Detection](https://www.kaggle.com/datasets/hemanthsai7/solar-panel-dust-detection) — hemanthsai7 on Kaggle

Downloaded in the notebook via:

```python
import kagglehub
path = kagglehub.dataset_download("hemanthsai7/solar-panel-dust-detection")
```

Split applied to the `Detect_solar_dust` source directory (seed = 36):

| Split | Clean | Dusty | Total |
|---|---|---|---|
| Train (80%) | 1,194 | 855 | 2,049 |
| Val (10%) | 149 | 106 | 255 |
| Test (10%) | 150 | 108 | 258 |

---

## Models

### Custom CNN

Three-block convolutional network built from scratch.

| Block | Filters | Components |
|---|---|---|
| 1 | 32 | Conv2D × 2, BatchNorm, MaxPool, Dropout(0.25) |
| 2 | 64 | Conv2D × 2, BatchNorm, MaxPool, Dropout(0.25) |
| 3 | 128 | Conv2D × 2, BatchNorm, MaxPool, Dropout(0.25) |
| Head | — | GAP → Dense(256, relu) → Dropout(0.5) → Dense(1, sigmoid) |

### MobileNetV3Small — Transfer Learning (Feature Extraction)

ImageNet-pretrained MobileNetV3Small with all 157 base layers frozen. Only the classification head is trained.

```
MobileNetV3Small (frozen) → GAP → Dense(256, relu) → Dropout(0.2) → Dense(1, sigmoid)
```

Preprocessing: `tf.keras.applications.mobilenet_v3.preprocess_input`

---

## Training

| Hyperparameter | Value |
|---|---|
| Input size | 224 × 224 × 3 |
| Batch size | 36 |
| Max epochs | 35 |
| Optimizer | Adam |
| Learning rate | 1e-4 |
| Loss | Binary crossentropy |
| Seed | 36 |

**Data augmentation** (training only):

- Random horizontal flip
- Random rotation (factor 0.2)
- Random zoom (factor 0.2)

**Callbacks** (shared by both models):

- `ModelCheckpoint` — saves best `val_accuracy` checkpoint
- `EarlyStopping` — `val_loss`, patience = 8, restores best weights
- `ReduceLROnPlateau` — `val_loss`, factor = 0.3, patience = 4, min lr = 1e-7

---

## Evaluation

### MobileNetV3Small — Test Set

```
              precision    recall  f1-score   support

       Clean     0.8395    0.9067    0.8718       150
       Dusty     0.8542    0.7593    0.8039       108

    accuracy                         0.8450       258
   macro avg     0.8468    0.8330    0.8379       258
weighted avg     0.8456    0.8450    0.8434       258
```

Confusion matrix:

```
                 Predicted
              Clean   Dusty
Actual Clean  [ 136      14 ]
       Dusty  [  26      82 ]
```

### Custom CNN — Test Set

```
              precision    recall  f1-score   support

       Clean     0.8710    0.7200    0.7883       150
       Dusty     0.6866    0.8519    0.7603       108

    accuracy                         0.7752       258
   macro avg     0.7788    0.7859    0.7743       258
weighted avg     0.7938    0.7752    0.7766       258
```

---

## Inference Logic

Both models output a single sigmoid neuron. `prob` represents P(Dusty) and the decision threshold is 0.5:

```python
prob  = model.predict(img_array, verbose=0)[0][0]
label = CLASS_NAMES[int(prob >= 0.5)]  # ["Clean", "Dusty"]
```

`CLASS_NAMES = ["Clean", "Dusty"]` must match the alphabetical directory order that `image_dataset_from_directory` assigns during training.

---

## Running the App

Install dependencies:

```bash
pip install streamlit tensorflow numpy
```

Place `tl_feature_extraction_best.keras` in the same directory as `app.py`, then run:

```bash
streamlit run app.py
```

Upload a panel image (JPG, PNG, BMP, or WEBP). The app returns the predicted class, raw sigmoid score, per-class probabilities, and a visual confidence gauge.
