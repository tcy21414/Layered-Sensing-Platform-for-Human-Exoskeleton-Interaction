# Torque Regression with IMU and EMG Data

## 🎯 Overview

The project aims to predict torque values by analyzing:
- **IMU Data**: Inertial measurement unit data processed through Temporal Convolutional Network (TCN)
- **EMG Data**: Electromyography signals processed through SENet
- **Multi-modal Fusion**: Combined analysis for improved prediction accuracy

## 🧪 Dataset

### Training Data
- **File**: `Estimation_Dataset.csv`
- **Content**: Multi-subject dataset with 8 subjects
- **Features**: 
  - 18 IMU channels (`IMU_1` to `IMU_18`)
  - 3 EMG channels (`EMG_1` to `EMG_3`)
  - Target: `ankle_moment` (torque values)

### Testing Data
- **File**: `Estimation_LOSO.csv`
- **Content**: Leave-one-subject-out (LOSO) validation dataset
- **Purpose**: Evaluation on an unseen subject for generalization testing


## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

