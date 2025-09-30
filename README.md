# EG-TransAttention: Edge-aware Multiscale Attention Network for Predicting circRNA-Disease Associations

## Description
EG-TransAttention is a deep learning framework designed to predict potential associations between circular RNAs (circRNAs) and diseases. It integrates edge-aware multiscale attention mechanisms to capture both local and global topological features from heterogeneous biological networks. The model combines graph convolutional operations with edge-guided attention to enhance representation learning and improve prediction accuracy.

## Dataset Information
- **Source**: circR2Disease and circRNADisease databases  
- **Format**: circRNA-disease association matrices, similarity networks, and node features  
- **Preprocessing**:  
  - Construction of circRNA and disease similarity networks  
  - Integration into a heterogeneous graph  
  - Normalization and train-test split

## Requirements
- Python ≥ 3.8  
- PyTorch ≥ 1.10  
- NumPy  
- SciPy  
- scikit-learn  
- NetworkX  
- tqdm

## Code Structure
- `main.py`: Entry point for training and evaluation  
- `train1.py`: Training loop and model optimization  
- `model.py`: Defines EG-TransAttention architecture and attention modules  
- `load_data.py`: Loads and preprocesses circRNA-disease data  
- `param.py`: Contains hyperparameters and configuration settings  
- `evaluation_scores.py`: Computes evaluation metrics (AUC, accuracy, F1-score, etc.)  
- `MA.py` / `PA.py`: Implements multiscale attention and path-aware attention  
- `figure.py` / `pic.py`: Generates visualizations of results and attention maps
## Computing Environment
Runtime: Python 3.9, no GPU required

Note: All experiments were conducted on CPU. The model is lightweight and can be trained and evaluated without GPU acceleration.

## Reproducibility
- All random seeds can be set via `--seed` argument in `main.py`  
- Dataset and model configuration are defined in `param.py`  
- Evaluation metrics are computed using `evaluation_scores.py`  
