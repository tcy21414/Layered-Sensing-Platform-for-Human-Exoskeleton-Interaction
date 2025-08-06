# Ankle Danger Detection with SE-ResNet1D

This project implements a binary classifier to detect potential ankle injury risks using strain sensor signals. The model is based on a 1D Squeeze-and-Excitation ResNet and trained with Focal Loss to handle class imbalance.

## Features
- Input: 2-channel strain signals
- Architecture: SE-ResNet1D
- Loss: Focal Loss for danger state emphasis
- Evaluation: Accuracy and Danger Recall
- Test mode: LOSO (Leave-One-Subject-Out) Zero-shot
