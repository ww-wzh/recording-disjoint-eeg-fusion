# Recorded software and hardware environment

The audited run recorded:

- Windows 11
- Python 3.12.13
- NumPy 2.4.4
- pandas 3.0.2
- SciPy 1.17.1
- scikit-learn 1.8.0
- matplotlib 3.10.9
- PyTorch 2.9.0 with CUDA 12.9
- pyRiemann 0.11
- NVIDIA GeForce RTX 5060 Laptop GPU

Random seeds were 1335, 1388, 1441, 1494 and 1547. PyTorch used `cudnn.deterministic=True` and `cudnn.benchmark=False`. GPU and library differences can change fitted probabilities slightly; the included frozen predictions and hashes define the reported result.

Install the ordinary Python dependencies with `requirements.txt`. Install a PyTorch build appropriate for the destination machine by following the official PyTorch installation selector; CUDA wheels are platform-specific.
