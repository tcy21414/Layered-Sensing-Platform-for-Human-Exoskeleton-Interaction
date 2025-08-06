# Metabolic State Classification using IMU and EMG Signals

This repository implements a deep learning framework for classifying metabolic states based on synchronized multimodal wearable data. Specifically, it leverages 18-channel Inertial Measurement Unit (IMU) signals and 6-channel Electromyography (EMG) signals to capture motion dynamics and muscle activation patterns. The model architecture combines a Temporal Convolutional Network (TCN) for processing IMU data with a Squeeze-and-Excitation ResNet (SE-ResNet) for EMG signal encoding.

> 🧠 The system is designed to operate on 9-second samples, consisting of a 3-second fixed context and a 6-second sliding window (9000 timepoints @ 1000 Hz).
