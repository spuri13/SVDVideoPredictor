
# SVD-Based Video Frame Predictor with Custom LSTM

## Overview

This project implements a video frame predictor that combines **Singular Value Decomposition (SVD)** for compression and a **from-scratch LSTM-like architecture** for temporal learning and future frame prediction. It demonstrates both an understanding of classical linear algebra techniques and deep learning principles — built entirely using NumPy without relying on high-level neural network libraries for the model itself.

## Project Structure

### 1. **Frame Extraction and Compression**
- Extracts frames from a grayscale video.
- Applies **SVD** to each frame, reducing dimensionality to a rank-`r` approximation.
- Flattens and concatenates the truncated `U`, `Σ`, and `Vᵗ` matrices into a 1D vector representation of the frame.

### 2. **Sequence Construction**
- Frames are grouped into input-output pairs using a sliding window (`sequence_num` context frames used to predict the next).
- Each windowed sequence is fed into a custom LSTM-like model.

### 3. **Custom LSTM Implementation**
- Implements core LSTM components from scratch using NumPy:
  - **Forget gate**
  - **Input gate**
  - **Candidate (potential) memory**
  - **Output gate**
- Learns to update hidden and cell states over the sequence to predict the next frame's compressed vector.

### 4. **Prediction and Reconstruction**
- The final output vector (predicted frame) is reconstructed using the inverse SVD process:
  - `U_r @ diag(S_r) @ Vt_r`
- The predicted frame is visualized alongside the ground truth.

## Why This Matters

This project:
- Emphasizes **low-level understanding** of LSTM mechanics.
- Leverages **video compression** as a form of feature extraction.
- Is useful for domains like **video forecasting**, **surveillance**, and **sports analytics** — where predicting next frames is critical.

## Results

- Loss decreases steadily over epochs (can be extended for better performance).
- Generated frames are structurally aligned with actual future frames, though clarity may vary with training length and SVD rank.
- Tunable parameters: `r` (SVD rank), `hidden_size`, `sequence_num`, and epochs.

## Technologies Used

- **Python**, **NumPy**, **OpenCV**, **Matplotlib**
- No deep learning libraries (except optional PyTorch for testing a dense layer baseline)

---

## Usage Notice

**Please do not copy this code directly for submission or job applications without permission.**  
This project was built for educational, portfolio, and demonstration purposes — and reflects a significant amount of original effort and understanding.

If you’re inspired by this project, feel free to fork it and **credit appropriately**. Thanks!

---

## Contact

**Samarth Puri**  
Computational Data Science @ Penn State 