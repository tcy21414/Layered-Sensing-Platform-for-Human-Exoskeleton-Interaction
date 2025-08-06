# Torque Regression with IMU and EMG Data

## 🎯 Overview

The project aims to predict torque values by analyzing:
- **IMU Data**: Inertial measurement unit data processed through Temporal Convolutional Network (TCN)
- **EMG Data**: Electromyography signals processed through SENet
- **Multi-modal Fusion**: Combined analysis for improved prediction accuracy

## 🚀 Quick Start

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/JINLuyao9/torque_regression.git
   cd torque_regression
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the main script:
   ```bash
   python main.py
   ```

## 🧪 Dataset

### Training Data
- **File**: `Estimation_dataset.csv`
- **Content**: Multi-subject dataset with 8 subjects
- **Features**: 
  - 18 IMU channels (`IMU_1` to `IMU_18`)
  - 3 EMG channels (`EMG_1` to `EMG_3`)
  - Target: `ankle_moment` (torque values)

### Testing Data
- **File**: `Estimation_LOSO.csv`
- **Content**: Leave-one-subject-out (LOSO) validation dataset
- **Purpose**: Evaluation on an unseen subject for generalization testing

## 📝 Citation

If you use this code in your research, please cite:
```bibtex
@misc{torque_regression,
  author = {},
  title = {Torque Regression with IMU and EMG Data using TCN and SENet},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/JINLuyao9/torque_regression}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
