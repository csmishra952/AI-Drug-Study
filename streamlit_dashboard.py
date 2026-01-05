import streamlit as st
import pandas as pd
import torch
import altair as alt
from model import GNN
from utils import smiles_to_graph, get_3d_block
from stmol import showmol
import py3Dmol
st.set_page_config(page_title="AI Drug Discovery", layout="wide")
st.title("AI Drug Discovery & Graph Neural Networks")

# Sidebar: Inputs
st.sidebar.header("Input Molecule")
smiles_input = st.sidebar.text_input("Enter SMILES String", value="CC(=O)OC1=CC=CC=C1C(=O)O") # Default: Aspirin
st.sidebar.markdown("**Examples:**")
st.sidebar.code("CN1C=NC2=C1C(=O)N(C(=O)N2C)C") # Caffeine
st.sidebar.code("CC1=C(C=C(C=C1)O)C(=O)O") # Paracetamol

# Main Layout
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("1. 3D Molecular Structure")
    blk = get_3d_block(smiles_input)
    if blk:
        view = py3Dmol.view(width=600, height=400)
        view.addModel(blk, 'mol')
        view.setStyle({'stick': {}})
        view.setBackgroundColor('white')
        view.zoomTo()
        showmol(view, height=400, width=600)
    else:
        st.error("Invalid SMILES string or could not generate 3D structure.")

    st.subheader("2. AI Property Prediction")
    if st.button("Predict Solubility"):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = GNN(hidden_channels=64, num_node_features=9) 
        try:
            model.load_state_dict(torch.load('drug_discovery_model.pt', map_location=device))
            model.eval()
            graph_data = smiles_to_graph(smiles_input)
            
            if graph_data:
                batch = torch.zeros(graph_data.x.shape[0], dtype=torch.long) 
                with torch.no_grad():
                    pred = model(graph_data.x, graph_data.edge_index, batch)
                
                val = pred.item()
                st.success(f"Predicted Log Solubility (ESOL): {val:.4f} mol/L")
                if val < -4: st.info("📉 Low Solubility (Lipophilic)")
                elif val > -2: st.info("QM Water Soluble (Hydrophilic)")
                else: st.info("⚖️ Moderate Solubility")
            else:
                st.error("Could not process molecule graph.")
        except FileNotFoundError:
            st.warning("Model file not found. Please run 'train.py' first!")

    st.subheader("3. Model Training History")
    try:
        df = pd.read_csv("train_history.csv")
        chart = alt.Chart(df).mark_line(point=True, color='purple').encode(
            x='epoch',
            y='loss',
            tooltip=['epoch', 'loss']
        ).properties(title='Training Loss Curve')
        st.altair_chart(chart, use_container_width=True)
    except FileNotFoundError:
        st.write("No training history found.")