# EG-TransAttention: Edge-aware Multiscale Attention Network for Predicting circRNA-Disease Associations

## Description
EG-TransAttention is a deep learning framework designed to predict potential associations between circular RNAs (circRNAs) and diseases. It integrates edge-aware multiscale attention mechanisms to capture both local and global topological features from heterogeneous biological networks. The model combines graph convolutional operations with edge-guided attention to enhance representation learning and improve prediction accuracy.

---
## Dataset Information

The `datasets` folder contains the processed circRNA–disease association data, similarity matrices, and name mapping files used for training and evaluation.  

### Files Description

| File Name                        | Description                                                                 |
|-----------------------------------|-----------------------------------------------------------------------------|
| `all_circrna_disease_pairs.csv`   | All known circRNA–disease associations used as the base dataset.            |
| `c_c.csv`                         | circRNA–circRNA similarity matrix (e.g., based on sequence or functional similarity). |
| `c_d.csv`                         | circRNA–disease association matrix (binary: 1 for known association, 0 for unknown). |
| `c_d_name.csv`                    | circRNA–disease pair names corresponding to `c_d.csv` indices.              |
| `circname.txt`                    | List of circRNA names corresponding to the circRNA similarity matrices.     |
| `d_d.csv`                         | disease–disease similarity matrix (e.g., based on semantic similarity).     |
| `disease semantic similarity.csv` | Additional disease semantic similarity scores used for constructing graphs. |
| `disname.txt`                     | List of disease names corresponding to the disease similarity matrices.     |
| `funicircRNA.csv`                 | Functional similarity information of circRNAs.                             |
| `GCC_similarity.csv`             | Global topological similarity of circRNAs.                                 |
| `GDD_similarity.csv`             | Global topological similarity of diseases.                                 |

### Data Source
The data is collected and integrated from the **circR2Disease** and **circRNADisease** databases, then processed into:
- circRNA–circRNA similarity matrices
- disease–disease similarity matrices
- circRNA–disease association matrices
- circRNA/disease name mapping files

These files are loaded automatically by `load_data.py` during training.


## Code Structure

- `main.py`: Entry point for training and evaluation  
- `train1.py`: Training loop and model optimization  
- `model.py`: Defines EG-TransAttention architecture and attention modules  
- `load_data.py`: Loads and preprocesses circRNA-disease data  
- `param.py`: Contains hyperparameters and configuration settings  
- `evaluation_scores.py`: Computes evaluation metrics (AUC, accuracy, F1-score, etc.)  
- `MA.py` / `PA.py`: Implements multiscale attention and Parnet attention  
 
## Usage Instructions
- Prepare the Dataset

  Download circRNA–disease association data from circR2Disease and circRNADisease.

- Generate Fusion Similarities

  Before training, run load_data.py to preprocess the data and compute Gaussian-fused similarities:

  python load_data.py

- This will:

  Read raw matrices: c_d.csv, c_c.csv, d_d.csv.

- Compute Gaussian similarity matrices:

  GCC_similarity.csv for circRNAs

  GDD_similarity.csv for diseases

  Integrate original similarity matrices with Gaussian similarities.

  Prepare inputs for graph construction in the model.

⚠️ Make sure the paths in param.py match your dataset location.

- Set Hyperparameters

  Edit param.py to configure model parameters, such as learning rate, epochs, batch size, number of attention heads, etc.

- Run Training
  python main.py

  Loads preprocessed data from load_data.py

  Constructs heterogeneous graphs

  Trains the EG-TransAttention model

  Outputs prediction scores and learned node features


## Requirements
- Python ≥ 3.8  
- PyTorch ≥ 1.10  
- NumPy  
- SciPy  
- scikit-learn  
- NetworkX  
- tqdm  

