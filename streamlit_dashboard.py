import streamlit as st
import pandas as pd
import torch
import altair as alt
import os
from model import GNN
from utils import smiles_to_graph, get_3d_block
import py3Dmol
st.set_page_config(
    page_title="AI Drug Discovery",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "AI Drug Discovery using Graph Neural Networks"}
)

st.markdown("""
    <style>
    /* Main background */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
    }
    
    /* Header styling */
    .header-title {
        text-align: center;
        color: white;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .header-subtitle {
        text-align: center;
        color: rgba(255,255,255,0.9);
        font-size: 1.1em;
        margin-bottom: 30px;
    }
    
    /* Card styling */
    .card {
        background: white;
        border-radius: 12px;
        padding: 25px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        margin-bottom: 20px;
    }
    
    .card-title {
        font-size: 1.4em;
        font-weight: bold;
        color: #667eea;
        margin-bottom: 15px;
        border-left: 4px solid #667eea;
        padding-left: 10px;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 30px;
        font-weight: bold;
        transition: transform 0.2s;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: scale(1.02);
    }
    
    /* Metric styling */
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    
    .metric-value {
        font-size: 2em;
        font-weight: bold;
    }
    
    .metric-label {
        font-size: 0.9em;
        opacity: 0.9;
        margin-top: 5px;
    }
    
    /* Sidebar */
    .sidebar-content {
        background: rgba(255,255,255,0.95);
        border-radius: 12px;
        padding: 20px;
    }
    </style>
""", unsafe_allow_html=True)
col_header = st.columns([1])
with col_header[0]:
    st.markdown('<div class="header-title">💊 AI Drug Discovery</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-subtitle">Graph Neural Networks for Molecular Property Prediction</div>', unsafe_allow_html=True)
st.markdown("---")
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.divider()
    
    st.markdown("#### 🧪 Input Molecule")
    smiles_input = st.text_input(
        "Enter SMILES String",
        value="CC(=O)OC1=CC=CC=C1C(=O)O",
        help="Standard SMILES notation for chemical compounds"
    )
    
    st.markdown("#### 📚 Example Molecules")
    examples = {
        "Caffeine ☕": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "Paracetamol 💊": "CC1=C(C=C(C=C1)O)C(=O)O",
        "Aspirin 🩺": "CC(=O)OC1=CC=CC=C1C(=O)O"
    }
    
    for name, smiles in examples.items():
        if st.button(f"Load {name}", use_container_width=True):
            smiles_input = smiles
            st.rerun()
    
    st.divider()
    st.markdown("### 📖 About")
    st.info("""
    **AI Drug Discovery** uses Graph Convolutional Networks (GCN) to predict 
    molecular solubility from chemical structures.
    
    **Dataset:** ESOL (1128 compounds)
    **Model:** 3-layer GCN with 64 hidden channels
    """)

tab1, tab2, tab3 = st.tabs([" Analysis", " Model Info", " Training Metrics"])

with tab1:
    st.markdown('<div class="card"><div class="card-title">🧬 Molecular Structure Visualization</div>', unsafe_allow_html=True)
    
    col_mol = st.columns([2, 1])
    
    with col_mol[0]:
        blk = get_3d_block(smiles_input)
        if blk:
            html_str = f"""
            <div id="viewer" style="width: 100%; height: 500px; position: relative; border-radius: 8px; overflow: hidden;"></div>
            <script src="https://3Dmol.csb.pitt.edu/build/3Dmol-min.js"></script>
            <script>
                let viewer = $3Dmol.createViewer("viewer", {{backgroundColor: '#f0f0f0'}});
                let pdbData = `{blk}`;
                viewer.addModel(pdbData, "mol");
                viewer.setStyle({{}}, {{stick: {{colorscheme: 'Jmol'}}}});
                viewer.zoomTo();
                viewer.render();
            </script>
            """
            st.components.v1.html(html_str, height=520)
        else:
            st.error(" Invalid SMILES string. Please check the input.")
    
    with col_mol[1]:
        st.markdown("####  SMILES Info")
        st.code(smiles_input, language="text")
        st.markdown("---")
        st.markdown("####  Tips")
        st.markdown("""
        - Valid SMILES notation required
        - Use standard chemical notation
        - Check examples in sidebar
        """)
    
    st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="card"><div class="card-title"> Model Architecture</div>', unsafe_allow_html=True)
    
    col_arch = st.columns(3)
    
    with col_arch[0]:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-value">3</div>
            <div class="metric-label">GCN Layers</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_arch[1]:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-value">64</div>
            <div class="metric-label">Hidden Channels</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_arch[2]:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-value">1</div>
            <div class="metric-label">Output (Regression)</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    st.markdown("####  Model Details")
    st.markdown("""
    **Graph Convolutional Network (GCN)**
    - Processes molecular graphs as input
    - Learns atomic and bond relationships
    - Outputs Log Solubility (ESOL scale)
    
    **Input Features (per atom):**
    - Atomic number
    - Degree
    - Formal charge
    - Radical electrons
    - Hybridization state
    - Aromaticity
    - Hydrogen count
    """)
    
    st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="card"><div class="card-title"> Training Performance</div>', unsafe_allow_html=True)
    
    try:
        df = pd.read_csv("train_history.csv")
        
        col_stats = st.columns(3)
        with col_stats[0]:
            st.metric("Total Epochs", int(df['epoch'].max()))
        with col_stats[1]:
            st.metric("Final Loss", f"{df['loss'].iloc[-1]:.6f}")
        with col_stats[2]:
            st.metric("Best Loss", f"{df['loss'].min():.6f}")
        
        st.divider()
        chart = alt.Chart(df).mark_line(point=True, size=3).encode(
            x=alt.X('epoch:Q', title='Epoch', scale=alt.Scale(zero=False)),
            y=alt.Y('loss:Q', title='Loss (MSE)', scale=alt.Scale(zero=False)),
            tooltip=['epoch:Q', alt.Tooltip('loss:Q', format='.6f')]
        ).properties(
            title='Training Loss Curve',
            height=400,
            width=800
        ).interactive()
        
        st.altair_chart(chart, width='stretch')
        
    except FileNotFoundError:
        st.warning(" No training history found. Run `python train.py` first.")
    
    st.markdown("</div>", unsafe_allow_html=True)
st.markdown("---")
st.markdown('<div class="card"><div class="card-title"> AI Property Prediction</div>', unsafe_allow_html=True)

col_pred = st.columns([1, 2])

with col_pred[0]:
    if st.button(" Predict Solubility", use_container_width=True):
        with st.spinner(" Running prediction..."):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = GNN(hidden_channels=64, num_node_features=9)
            
            try:
                if os.path.exists('drug_discovery_model.pt'):
                    model.load_state_dict(torch.load('drug_discovery_model.pt', map_location=device))
                else:
                    st.warning(" Model weights not found. Using untrained network.")
                
                model.eval()
                graph_data = smiles_to_graph(smiles_input)
                
                if graph_data:
                    batch = torch.zeros(graph_data.x.shape[0], dtype=torch.long)
                    with torch.no_grad():
                        pred = model(graph_data.x, graph_data.edge_index, batch)
                    
                    val = pred.item()
                    if val < -4:
                        category = " Low Solubility (Lipophilic)"
                        color = "#d32f2f"
                    elif val > -2:
                        category = " High Solubility (Hydrophilic)"
                        color = "#388e3c"
                    else:
                        category = " Moderate Solubility"
                        color = "#f57c00"
                    
                    st.markdown(f"""
                    <div style="background: {color}; color: white; padding: 20px; border-radius: 10px; text-align: center;">
                        <div style="font-size: 2.5em; font-weight: bold;">{val:.4f}</div>
                        <div style="font-size: 1.2em; margin-top: 10px;">Log Solubility (mol/L)</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"**Category:** {category}")
                else:
                    st.error(" Could not process molecule graph.")
            except Exception as e:
                st.error(f" Prediction Error: {str(e)}")
                st.info("Ensure 'train.py' has been executed to generate model weights.")

with col_pred[1]:
    st.markdown("####  Solubility Scale (ESOL)")
    st.markdown("""
    | Range | Category | Meaning |
    |-------|----------|---------|
    | **< -4** |  Low | Lipophilic, poor water solubility |
    | **-4 to -2** |  Moderate | Balanced properties |
    | **> -2** |  High | Hydrophilic, good water solubility |
    **Use Case:** Drug candidates typically need moderate to high solubility.
    """)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9em;">
    <p> AI Drug Discovery Platform | Built with Streamlit, PyTorch & RDKit</p>
    <p>Dataset: ESOL (Delaney Solubility) | Model: Graph Convolutional Network</p>
</div>
""", unsafe_allow_html=True)