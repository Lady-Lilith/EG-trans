# EG-TransAttention: Edge-aware Multiscale Attention Network for Predicting circRNA-Disease Associations

## Description
EG-TransAttention is a deep learning framework designed to predict potential associations between circular RNAs (circRNAs) and diseases. It integrates edge-aware multiscale attention mechanisms to capture both local and global topological features from heterogeneous biological networks. The model combines graph convolutional operations with edge-guided attention to enhance representation learning and improve prediction accuracy.

---

## Dataset Information
- **Source**: circR2Disease and circRNADisease databases  
- **Format**: circRNA-disease association matrices, similarity networks, and node features  
- **Preprocessing**:  
  - Construction of circRNA and disease similarity networks  
  - Integration into a heterogeneous graph  
  - Normalization and train-test split  

---

## Requirements
- Python ≥ 3.8  
- PyTorch ≥ 1.10  
- NumPy  
- SciPy  
- scikit-learn  
- NetworkX  
- tqdm  

Install dependencies with:
```bash
pip install -r requirements.txt
