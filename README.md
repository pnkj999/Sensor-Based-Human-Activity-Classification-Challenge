# Sensor-Based-Human-Activity-Classification-Challenge
The PIRvision dataset contains occupancy detection data collected using a synchronized Low- Energy Electronically-chopped Passive Infra-Red (PIR) sensor node deployed in residential and  office environments. Each observation corresponds to 4 seconds of recorded human activity within the sensor’s Field-of-View (FoV).Evalueate and classify 
# PIRVision Office Dataset Analysis & Modeling

##  Dataset Overview

The PIRVision Office Dataset contains motion sensor readings from multiple PIR (Passive Infrared) sensors in an office environment. The goal is to detect human presence and activity.

### Label Definitions:

- **Label 0**: No activity (vacant)
- **Label 1**: Stationary human presence
- **Label 3**: Movement or anomaly (possibly sensor error or transient motion)

---

##  Exploratory Data Analysis (EDA)

### Key EDA Tasks:

1. **Missing Values**: Checked using `df.isnull().sum()` — No missing values.
2. **Label Distribution**: Highly imbalanced — label 0 dominates.
3. **Sensor Insights**:
   - `PIR_1` had extreme outliers (> 20,000), replaced with 0.
   - Top 5 active sensors visualized using pie chart.
4. **Temperature Correlation**:
   - Unusual correlation between low temp (~0°F) and label 3.
5. **Feature Independence**:
   - Correlation heatmap showed mostly independent features.
6. **Autocorrelation Analysis**:
   - Labels show no strong temporal dependency.

---

##  Modeling

### Preprocessing:

- Standardization using `StandardScaler`
- Label encoding with `LabelEncoder`
- SMOTE for balancing classes



##  Model 1: Multi-Layer Perceptron (MLP)

###  Architecture

- `Fully Connected (Linear) → ReLU → Dropout`
- `Fully Connected (Linear) → ReLU → Dropout`
- `Final Output Layer` with `CrossEntropyLoss` for multi-class classification

---

###  Libraries Used

- **Deep Learning**: `torch`, `torch.nn`, `torch.optim`
- **Data Processing & Evaluation**: `sklearn` (metrics, model_selection, preprocessing)
- **Class Balancing**: `imblearn.over_sampling.SMOTE`
- **Visualization**: `matplotlib`, `seaborn`
- **Data Handling**: `numpy`, `pandas`

---

###  Code Workflow

1. **Data Preparation**:
   - Missing values handled and outliers processed (if any)
   - Categorical labels encoded using `LabelEncoder`
   - Feature scaling done via `StandardScaler`
   - No reshaping required (unlike CNN)

2. **Class Balancing**:
   - Applied `SMOTE` to oversample minority classes before model training

3. **Model Training**:
   - Defined a custom MLP using PyTorch
   - Included 2 fully connected layers with dropout
   - Optimizers evaluated using custom training loop

4. **Training Enhancements**:
   - Early Stopping based on validation loss
   - `ReduceLROnPlateau` for learning rate adjustment
   - Best models saved using checkpointing

5. **Performance Evaluation**:
   - Stratified 5-Fold Cross Validation
   - Metrics tracked: Accuracy, Precision, Recall, F1 Score, Convergence Epoch
   - Visualized loss landscape and optimizer parameter trajectories



###  Results Summary

####  Optimizer Comparison


| Optimizer        | Accuracy | Precision | Recall  | F1-score |
|------------------|----------|-----------|---------|----------|
| Adam             | 0.9656   | 0.9203    | 0.9789  | 0.9449   |
| Adam (AMSGrad)   | 0.9601   | 0.9106    | 0.9785  | 0.9382   |
| NAG              | 0.9025   | 0.8436    | 0.9588  | 0.8761   |
| SGD              | 0.6485   | 0.5217    | 0.5727  | 0.4478   |
| SGD + Momentum   | 0.9178   | 0.8545    | 0.9645  | 0.8894   |

---


####  Learning Rate Tuning with Adam (Coarse Search)

| Learning Rate | Accuracy | F1 Score | Convergence Epoch |
|---------------|----------|----------|-------------------|
| 0.01          | 0.9881   | 0.9784   | 50.6              |
| 0.001         | 0.9601   | 0.9382   | 100               |
| 0.1           | 0.5379   | 0.4072   | 33.2              |
| 0.0001        | 0.6140   | 0.5318   | 100               |
| 0.00001       | 0.5202   | 0.3149   | 92.2              |

---

###  Inference & Evaluation

To evaluate your model on new data using a trained checkpoint, use the `evaluate_model()` function:

```python
def evaluate_model(datafile_location, checkpoint_path):
    ...
```

This function:
- Loads and preprocesses the given CSV data (standard scaling + label encoding)
- Loads the saved model checkpoint
- Evaluates accuracy, precision, recall, F1 score, confusion matrix, and classification report

Example usage:

```python
evaluate_model("new_data.csv", "best_model.pt")
```

Outputs include:
- Detailed per-class metrics
- Confusion matrix and classification report
- Checkpoint-level metrics (if available)

---

###  Final Notes

- After tuning, **Adam with learning rate = 0.01** provided the best performance in both speed and accuracy.
- The MLP was effective for tabular data, especially when enhanced with dropout and learning rate scheduling.
- SMOTE significantly boosted recall for minority classes.
- `evaluate_model` provides a seamless way to test models on real-world or unseen datasets.


##  Model 2: 1D Convolutional Neural Network (1D CNN)

###  Architecture

- `Conv1D → ReLU → MaxPool → Conv1D → ReLU → MaxPool`
- `Fully Connected (Linear) → ReLU → Dropout`
- `Final Output Layer` with `CrossEntropyLoss` for multi-class classification

---

###  Libraries Used

- **Deep Learning**: `torch`, `torch.nn`, `torch.optim`
- **Data Processing & Evaluation**: `sklearn` (metrics, model_selection, preprocessing)
- **Class Balancing**: `imblearn.over_sampling.SMOTE`
- **Visualization**: `matplotlib`, `seaborn`
- **Data Handling**: `numpy`, `pandas`

---

### Code Workflow

1. **Data Preparation**:
   - Missing values handled and outliers processed (if any)
   - Categorical labels encoded using `LabelEncoder`
   - Feature scaling done via `StandardScaler`
   - Reshaping for Conv1D compatibility

2. **Class Balancing**:
   - Applied `SMOTE` to oversample minority classes before model training

3. **Model Training**:
   - Defined a custom 1D CNN using PyTorch
   - Included 2 Conv1D layers + FC layer with dropout
   - Optimizers evaluated using custom training loop

4. **Training Enhancements**:
   - Early Stopping based on validation loss
   - `ReduceLROnPlateau` for learning rate adjustment
   - Best models saved using checkpointing

5. **Performance Evaluation**:
   - Stratified 5-Fold Cross Validation
   - Metrics tracked: Accuracy, Precision, Recall, F1 Score, Convergence Epoch
   - Visualized loss landscape and optimizer parameter trajectories

---

###  Results Summary

#### Optimizer Comparison

| Optimizer        | Accuracy | Precision | Recall  | F1-score |
|------------------|----------|-----------|---------|----------|
| **Adam**         | 0.9850   | 0.9590    | 0.9907  | **0.9739** |
| Adam (AMSGrad)   | 0.9837   | 0.9558    | 0.9886  | 0.9712   |
| NAG              | 0.7984   | 0.8096    | 0.8571  | 0.7755   |
| SGD              | 0.6382   | 0.4901    | 0.5808  | 0.4573   |
| SGD + Momentum   | 0.3423   | 0.2813    | 0.5000  | 0.2835   |

#### 📉 Learning Rate Tuning with Adam (Coarse Search)

| Learning Rate | Accuracy | F1 Score | Convergence Epoch |
|---------------|----------|----------|-------------------|
| **0.01**      | 0.9865   | **0.9772** | 39.4              |
| 0.001         | 0.9846   | 0.9736   | 98.8              |
| 0.1           | 0.6153   | 0.5930   | 41.6              |
| 0.0001        | 0.4948   | 0.5289   | 64.4              |
| 0.00001       | 0.2442   | 0.1668   | 50.4              |

---

### Inference & Evaluation

To evaluate your model on new data using a trained checkpoint, use the `evaluate_model()` function:

```python
def evaluate_model(datafile_location, checkpoint_path):
    ...
```

This function:
- Loads and preprocesses the given CSV data (standard scaling + label encoding)
- Reshapes it for 1D CNN input
- Loads the saved model checkpoint
- Evaluates accuracy, precision, recall, F1 score, confusion matrix, and classification report

Example usage:

```python
evaluate_model("new_data.csv", "best_model.pt")
```

Outputs include:
- Detailed per-class metrics
- Confusion matrix and classification report
- Checkpoint-level metrics (if available)

---

### Final Notes

- After tuning, **Adam with learning rate = 0.01** gave the best results with a high F1-score and fast convergence.
- The CNN effectively captures temporal patterns in the 1D sequence data.
- SMOTE successfully addressed class imbalance, improving minority class performance.
- `evaluate_model` helps in testing the model on unseen real-world data with zero setup fuss.


##  Model 3: TabNet Classifier

### Why TabNet?

- Transformer-inspired architecture
- Handles tabular data without assuming temporal dependency

### Parameters (Default + Tweaks):

- `n_d = 8`, `n_a = 8`, `n_steps = 3`
- `optimizer = Adam / SGD`
- `lr = 1e-2 (Adam)` or `0.5 (SGD)`
- `batch_size = 1024 (Adam)` or `4096 (SGD)`

###  Libraries Used (TabNet):
- `pytorch-tabnet`
- `torch`
- `sklearn` (StratifiedKFold, preprocessing, metrics)
- `imblearn` (SMOTE)
- `pandas`, `numpy`
- `matplotlib`, `seaborn`, `TSNE`, `pickle`

###  How the Code Works:
1. Dataset loaded and label-encoded
2. SMOTE applied for class balancing
3. StandardScaler fitted and saved for each fold
4. TabNetClassifier is trained with specified hyperparameters
5. Evaluation via accuracy, precision, recall, F1-score
6. Confusion matrix and t-SNE plots generated
7. Best models saved as `.pth` and scalers/encoders saved as `.pkl`
8. Evaluation script reuses saved components for inference

### Training Instructions:
- First upload the `pirvision_office_dataset2.csv` dataset
- Run the first TabNet training code (with Adam) which produces:
  - 5 fold `.pth` weight files
  - `encoders.pkl` and `scalers.pkl`
- Use **these exact parameters** for inference:
"n_d": 8,
"n_a": 8,
"n_steps": 3,
"gamma": 1.5,
"lambda_sparse": 1e-4,
"optimizer_fn": torch.optim.Adam,
"optimizer_params": dict(lr=1e-2)
- During evaluation, upload: `encoders.pkl`, `scalers.pkl`, and one of the fold `.pth` files. Also specify `fold_num`.

### Second TabNet (SGD Variant):
- Repeat the above process, but change parameters:
"n_d": 4,
"n_a": 4,
"n_steps": 3,
"gamma": 1.5,
"lambda_sparse": 1e-3,
"optimizer_fn": torch.optim.SGD,
"optimizer_params": dict(lr=0.5, momentum=0.9)
- This run is more aggressively regularized, leading to underfitting.

### Results:

- TabNet with Adam outperformed SGD
- SGD with high LR + large batch caused underfitting (accuracy dropped to ~86%)

### TabNet Model Summary:
| Version           | Accuracy | Precision | Recall | F1-score |
| ----------------- | -------- | --------- | ------ | -------- |
| **TabNet + Adam** | 0.94     | 0.89      | 0.92   | 0.90     |
| **TabNet + SGD**  | 0.86     | 0.56      | 0.68   | 0.61     |

### TabNet Insights:

- TabNet is highly interpretable due to attention masks
- Good generalization and fast convergence
- Best with balanced data (SMOTE helps)
- Performance drops with aggressive regularization and high learning rate (SGD + LR=0.5)

---

##  Model Checkpointing & Evaluation

- Best models saved for each fold
- Final evaluation using saved models & standardized scalers
- Checkpoint restoration verified

###  Evaluation Script Notes:

- Requires: `scalers.pkl`, `encoders.pkl`, and model `.pth` file
- Parameters must match training configuration
- Accuracy reported per fold

---

## Insights & Takeaways

- PIR sensors are reliable, but outlier filtering is critical
- CNN and TabNet both perform well, with CNN slightly ahead
- Class imbalance significantly affects performance — SMOTE helped
- Adam remains the best optimizer
- Feature independence supports model robustness

---

##  Files Included:

- `training_code.py` — All model training logic
- `checkpoints/` — Saved best models per fold
- `scalers.pkl` / `encoders.pkl` — Preprocessing artifacts
- `optimizer_plots/` — Visualizations of loss landscape, metrics, and confusion matrices
- `encoders.pkl /` — LabelEncoder object used to encode labels during training and required during evaluation

---




