# AI Drug Discovery using Graph Neural Networks

A machine learning project for predicting molecular solubility using PyTorch Geometric and Graph Convolutional Networks (GCN).

## Features
- **GNN Model**: 3-layer Graph Convolutional Network for molecular property prediction
- **Interactive Dashboard**: Streamlit UI for real-time solubility predictions
- **SMILES Support**: Convert chemical notations to molecular graphs
- **3D Visualization**: py3Dmol integration for molecular structure viewing

## Installation

```bash
git clone https://github.com/csmishra952/AI-Drug-Study.git
cd AI-Drug-Study
pip install -r requirements.txt
```

## Usage

### Train the Model
```bash
python train.py
```

### Run Streamlit Dashboard
```bash
streamlit run streamlit_dashboard.py
```

The dashboard will open at `http://localhost:8501`

## Model Architecture
- **Input**: Molecular graphs (SMILES strings → atom features + edges)
- **GCN Layers**: 3 graph convolutional layers (64 hidden channels)
- **Output**: Log solubility prediction (regression)

## Dataset
- **ESOL**: Delaney Solubility Dataset (1128 compounds)
- **Location**: `esol/raw/delaney-processed.csv`

## Project Structure
```
├── model.py              # GNN architecture
├── train.py              # Training script
├── utils.py              # SMILES processing utilities
├── streamlit_dashboard.py # Interactive UI
├── train_history.csv     # Training metrics
└── esol/                 # Dataset directory
    ├── raw/
    └── processed/
```

## Technologies
- PyTorch & PyTorch Geometric
- RDKit (chemistry)
- Streamlit (UI)
- Pandas & Altair (visualization)