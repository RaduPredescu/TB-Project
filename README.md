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

Since it is a university project, we agreed NOT to have a branching strategy, which is a basic requirement in companies. So, all pushes were made directly on main.