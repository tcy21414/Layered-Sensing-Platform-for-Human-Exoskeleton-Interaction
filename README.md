# Layered-Sensing-Platform-for-Human-Exoskeleton-Interaction

This repository demonstrates the datasets and machine learning models code for the research manuscript:"A layered smart sensing platform for physiologically informed human-exoskeleton interaction"

## 🔍 Core Tasks

This system supports three key perception tasks critical for safe and adaptive human–exoskeleton interaction:

1. **Task I – Joint Moment Estimation**  
   Predicts the wearer’s joint torque, enabling intent-aware motion assistance.

2. **Task II – Metabolic Effort Monitoring**  
   Tracks real-time changes in metabolic energy consumption under varying assistive conditions.

3. **Task III – Injury Risk Detection**  
   Detects external mechanical overload or anomalous strain patterns to prevent unsafe torque application.

Each task is implemented as an independent neural module and evaluated under task-specific experimental conditions across lab, outdoor, and risk-triggered scenarios.


## 📁 Dataset Access

Due to GitHub’s file size limitations, all datasets are hosted externally on Google Drive:

https://drive.google.com/drive/folders/1FcqK_iQFRRlEzLQw-RtCexGELqiE9GPR?usp=sharing


For more details in the experiments, please drop a email to our corresponding authors.

## 📁 Exoskeleton Control

For the code to control the exoskeleton, please refer to (the control cycle is based on the gait cycle decoded by FSR data):

https://drive.google.com/drive/folders/1VYEdGYdG8HyOaK8iNNlknuvoAAej4QA2?usp=share_link

# ⚙️ System Requirements

### 🖥️ Operating Systems
- macOS 13.0 or later  
- Windows 10 or later  
- Ubuntu 20.04 or later  

### 📦 Software Dependencies
- **Python 3.8+**  
- **PyTorch 2.0.1** (for neural decoder training and inference)  
- **NumPy 1.24+**  
- **SciPy 1.10+**  
- **scikit-learn 1.2+**  
- **Matplotlib 3.7+**  
- **pandas 1.5+**  
- **tqdm 4.64+**  
- **CUDA 11.8+** (optional, for GPU acceleration)

### ⚡ Hardware Requirements
- Standard desktop/laptop (Intel i7/Ryzen 7 or Apple M1/M2 processor)  
- Recommended: NVIDIA GPU (RTX 3060 or higher, ≥8 GB VRAM) for efficient model training  
- Optional: Jetson Nano or ESP32 modules for embedded real-time deployment  

✅ Tested on:  
- macOS 14.1 (Apple M1 Max)  
- Ubuntu 22.04 (CUDA 12.1, RTX 4080)  
- Windows 11 (AMD Ryzen 7)
