# **💊 AI Drug Discovery: Solubility Prediction with GNNs & MLOps**

An end-to-end Machine Learning pipeline for predicting molecular solubility (LogS) using **Graph Attention Networks (GATv2)**. This project demonstrates modern AI engineering practices, including **MLOps** for experiment tracking, **Scaffold Splitting** for rigorous validation, and **Explainable AI (XAI)** to visualize atomic contributions.

## **Key Features**

* **Advanced Architecture**: Implements **GATv2 (Graph Attention Networks)** to learn dynamic relationships between atoms, outperforming standard GCNs.  
* **Deep Featurization**: Extracts 24 chemical features per atom (Hybridization, Aromaticity, Chirality) instead of simple atomic numbers.  
* **Explainable AI (XAI)**: Integrated **GNNExplainer** to visualize which atoms (e.g., Hydrophilic groups) drive the solubility prediction.  
* **MLOps Pipeline**: Built with **PyTorch Lightning** and **Weights & Biases (WandB)** for automated checkpointing, hyperparameter logging, and experiment tracking.  
* **Interactive Dashboard**: A Streamlit web app for real-time batch prediction and 3D molecular visualization.

## **Tech Stack**

* **Core**: PyTorch, PyTorch Geometric, RDKit  
* **Engineering**: PyTorch Lightning, WandB  
* **Visualization**: Py3Dmol, Streamlit, Altair  
* **Data**: ESOL (Delaney) Solubility Dataset

## **Installation**

git clone \[https://github.com/csmishra952/AI-Drug-Study.git\](https://github.com/csmishra952/AI-Drug-Study.git)  
cd AI-Drug-Study  
pip install \-r requirements.txt

## **Usage**

### **1\. Train the Model (MLOps Pipeline)**

This script handles featurization, scaffold splitting, and training with WandB logging.

python train\_pl.py

*Artifacts (best models) are automatically saved to checkpoints/.*

### **2\. Run the Dashboard**

Launch the interactive "Command Center" to analyze molecules.

streamlit run streamlit\_dashboard.py

## **MLOps & Performance**

The model is evaluated using **Scaffold Splitting**, a rigorous method that tests the model on structurally distinct molecules to ensure real-world generalization.

* **Training Framework**: PyTorch Lightning  
* **Experiment Tracking**: Weights & Biases (WandB)  
* **Early Stopping**: Monitored on Validation Loss

## **Explainability**

The project includes an XAI module that highlights atoms contributing to high or low solubility.

* **Red Atoms**: High contribution to the predicted property.  
* **Blue/White Atoms**: Neutral or negative contribution.
