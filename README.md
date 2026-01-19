# sEMG Movement Recognition – Project Pipeline

This project follows the requirements from the course assignment on surface EMG (sEMG) processing and movement recognition.  
It implements the full workflow described in the PDF:

1. Data loading  
2. Preprocessing  
3. Windowing  
4. Feature extraction  
5. Dataset splitting  

The goal is to transform raw sEMG recordings into a structured dataset suitable for machine learning experiments.

---

## Dataset

The project uses **EMG Database 1**, which contains sEMG recordings from the forearm during three flexion movements, performed by multiple subjects.

Each file has the format:

Subiect_<subject_id>_<class_id>_r.npy


where:
- `subjectId` identifies the participant
- `classId ∈ {0,1,2}` represents the movement class

Each file contains multiple EMG channels recorded simultaneously.

---

## Processing Pipeline

### 1. Data Loading
All `.npy` files are scanned and parsed.  
From each file, the EMG channels are extracted and the DC offset is removed.

Each file becomes one *example* with:
- a list of channels (signals)
- an associated movement label
- a subject identifier

---

### 2. Preprocessing

According to the course material, EMG signals must be cleaned and normalized before analysis.

For each channel:

1. **DC removal**  
   The mean value is subtracted to center the signal around zero.

2. **Artifact suppression**  
   Extreme outliers caused by motion artifacts are clipped using a robust MAD-based method.

3. **Normalization to maximum amplitude**  
   Each channel is normalized using:
   \[
   x_{norm} = \frac{x}{\max(|x|)}
   \]
   This follows the recommendation from the course to normalize EMG relative to a reference amplitude when MVIC is not available.

---

### 3. Windowing

The continuous EMG signals are segmented into fixed-length overlapping windows:

- Window length  
- Overlap

Each window has shape:


Every window inherits the label of the original recording.

This step converts long recordings into many short, stationary segments suitable for feature extraction.

---

### 4. Feature Extraction (Time Domain)

For each window and each channel, classical **time-domain EMG features** are computed, as required in the assignment:

- MAV – Mean Absolute Value  
- RMS – Root Mean Square  
- WL – Waveform Length  
- ZCR – Zero Crossing Rate (with threshold α)  
- SSC – Slope Sign Changes (with threshold α)  
- Skewness  
- ISEMG – Integrated Square-root EMG  
- Hjorth Activity  

With 8 channels and 8 features per channel, each window is represented by 64 features


This transforms raw signal windows into fixed-size numeric vectors suitable for machine learning.

---

### 5. Dataset Splitting

The dataset is split into:

- Training set  
- Validation set  
- Test set  

The split is performed **by subject**, not by window, in order to avoid data leakage.  
This ensures that windows from the same subject never appear in both training and testing sets.

The proportions are configurable in `config.yaml`, for example:

split:
  train: 0.70
  val: 0.15
  test: 0.15


**Important note!!!**

Since it was a university project, we agreed NOT to have a branching strategy, which is a basic requirement in companies. So, all pushes were made directly on main.

---

## MLP Experimental Results

| Nr | Epochs | Optimizer | Batch | LR   | LR sched | LR factor | Experiment        | Acc (%) | Prec (%) | Rec (%) | F1 (%) | Duration (s) |
|----|--------:|-----------|------:|------|----------:|-----------:|-------------------|--------:|---------:|--------:|-------:|-------------:|
| 1  | 100 | Adam | 256 | 1e-3 | 50  | 0.1 | 1st_experiment  | 80.20 | 80.25 | 80.20 | 80.15 | 67.59 |
| 2  | 200 | Adam | 256 | 1e-3 | 50  | 0.1 | 2nd_experiment  | 80.20 | 80.25 | 80.20 | 80.15 | 122.60 |
| 3  | 100 | Adam | 512 | 1e-3 | 50  | 0.1 | 3rd_experiment  | 79.06 | 79.30 | 79.06 | 79.13 | 44.33 |
| 4  | 200 | Adam | 512 | 1e-3 | 50  | 0.1 | 4th_experiment  | 79.06 | 79.30 | 79.06 | 79.13 | 88.23 |
| 5  | 200 | Adam | 256 | 1e-2 | 50  | 0.1 | 5th_experiment  | 65.88 | 64.71 | 65.88 | 64.86 | 121.40 |
| 6  | 200 | SGD  | 256 | 1e-3 | 50  | 0.1 | 6th_experiment  | 77.12 | 77.52 | 77.12 | 76.99 | 112.12 |
| 7  | 200 | SGD  | 128 | 1e-3 | 50  | 0.1 | 7th_experiment  | 78.61 | 78.64 | 78.61 | 78.58 | 197.01 |
| 8  | 200 | SGD  | 256 | 1e-3 | 150 | 0.1 | 8th_experiment  | 75.16 | 76.07 | 75.16 | 74.96 | 120.13 |
| 9  | 200 | SGD  | 256 | 1e-4 | 150 | 0.1 | 9th_experiment  | 77.75 | 77.81 | 77.75 | 77.70 | 133.13 |
| 10 | 400 | SGD  | 512 | 1e-3 | 200 | 0.1 | 10th_experiment | 77.45 | 78.26 | 77.45 | 77.49 | 178.87 |


---

## SVM Experimental Results

| Nr | C | Kernel | Acc (%) | Prec (%) | Rec (%) | F1 (%) | Confusion Matrix |
|----|---:|--------|--------:|---------:|--------:|-------:|------------------|
| 1  | 1 | rbf | 77.33 | 77.64 | 77.33 | 77.36 | [[1354, 426, 124], [245, 1475, 184], [115, 201, 1588]] |
| 2  | 2 | rbf | 77.78 | 77.97 | 77.78 | 77.77 | [[1363, 413, 128], [229, 1467, 208], [118, 173, 1613]] |
| 3  | 3 | rbf | 78.03 | 78.16 | 78.03 | 78.00 | [[1376, 402, 126], [229, 1461, 214], [122, 162, 1620]] |
| 4  | 4 | rbf | 78.15 | 78.26 | 78.15 | 78.14 | [[1387, 399, 118], [232, 1460, 212], [130, 157, 1617]] |
| 5  | 5 | rbf | 78.34 | 78.44 | 78.34 | 78.32 | [[1388, 397, 119], [228, 1466, 210], [131, 152, 1621]] |
| 6  | 1 | linear |  –  |  –  |  –  |  –  | – |
| 7  | 2 | linear |  –  |  –  |  –  |  –  | – |
| 8  | 3 | linear |  –  |  –  |  –  |  –  | – |
| 9  | 4 | linear |  –  |  –  |  –  |  –  | – |
| 10 | 5 | linear |  –  |  –  |  –  |  –  | – |
