import streamlit as st
import pandas as pd
import torch
import glob
import os
import numpy as np
from model_pl import DrugDiscoveryModel
from utils import smiles_to_graph, get_colored_3d_block, get_molecule_explanation
import streamlit.components.v1 as components
st.set_page_config(
    page_title="AI Drug Discovery Pro",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🧬"
)
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; font-weight: bold; }
    .highlight-box { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); }
    .prediction-card { text-align: center; padding: 20px; border-radius: 10px; color: white; }
    </style>
""", unsafe_allow_html=True)
@st.cache_resource
def load_model():
    checkpoints = glob.glob('checkpoints/*.ckpt')
    if not checkpoints:
        return None
    best_ckpt = checkpoints[-1] 
    model = DrugDiscoveryModel.load_from_checkpoint(best_ckpt)
    model.eval()
    return model

model = load_model()
col1, col2 = st.columns([1, 4])
with col1:
    st.markdown("<div style='font-size: 80px; text-align: center;'>🧬</div>", unsafe_allow_html=True)
with col2:
    st.title("AI Drug Discovery Platform")
    st.markdown("#### Explainable Graph Neural Networks (GATv2) for Solubility Prediction")

if not model:
    st.error("🚨 No model checkpoint found! Please run `python train_pl.py` first.")
    st.stop()
with st.sidebar:
    st.header("🧪 Input Molecule")
    input_method = st.radio("Choose Input:", ["Type SMILES", "Select Example"])
    
    if input_method == "Select Example":
        examples = {
            "Paracetamol (Painkiller)": "CC(=O)Nc1ccc(O)cc1",
            "Caffeine (Stimulant)": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            "Aspirin (Anti-inflammatory)": "CC(=O)Oc1ccccc1C(=O)O",
            "Lipophilic Chain (Low Sol)": "CCCCCCCCCCCCCCCC"
        }
        selected = st.selectbox("Choose a molecule:", list(examples.keys()))
        smiles_input = examples[selected]
    else:
        smiles_input = st.text_input("Enter SMILES:", "CC(=O)Nc1ccc(O)cc1")
    
    st.info(f"**Current Input:**\n`{smiles_input}`")
    st.markdown("---")
    st.markdown("### Model Stats")
    st.markdown("- **Architecture:** GATv2 (Attention)")
    st.markdown("- **Features:** 24 Deep Chemical Features")
    st.markdown("- **Training:** Scaffold Split (Rigorous)")
tab1, tab2, tab3 = st.tabs(["Analysis & XAI", " Batch Prediction", " Training Logs"])
with tab1:
    col_viz, col_res = st.columns([1.5, 1])
    graph_data = smiles_to_graph(smiles_input)
    
    if graph_data:
        with torch.no_grad():
            batch = torch.zeros(graph_data.x.shape[0], dtype=torch.long)
            pred_log_sol = model(graph_data.x, graph_data.edge_index, batch).item()
            
        sol_category = "High" if pred_log_sol > -2 else "Moderate" if pred_log_sol > -4 else "Low"
        color = "#2ecc71" if sol_category == "High" else "#f1c40f" if sol_category == "Moderate" else "#e74c3c"
        
        with col_res:
            st.markdown(f"""
            <div class="prediction-card" style="background: {color};">
                <h3>Predicted Log Solubility</h3>
                <h1 style="font-size: 3.5em;">{pred_log_sol:.3f}</h1>
                <p>Category: <b>{sol_category}</b> (mol/L)</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("###  AI Explanation")
            if st.button("✨ Explain Prediction (Why?)", type="primary"):
                with st.spinner("Analyzing atomic contributions..."):
                    importances = get_molecule_explanation(model, graph_data)
                    block, style_map = get_colored_3d_block(smiles_input, importances)
                    
                    st.session_state['viz_block'] = block
                    st.session_state['style_map'] = style_map
                    st.session_state['explained'] = True
            
            if st.session_state.get('explained'):
                st.success("Analysis Complete! Red atoms contribute most to the property.")

        with col_viz:
            st.markdown("### 🧬 3D Structure")
            if st.session_state.get('explained'):
                block = st.session_state['viz_block']
                style_map = st.session_state['style_map']
            else:
                block, _ = get_colored_3d_block(smiles_input)
                style_map = {}

            if block:
                style_json = str(style_map).replace("'", '"')
                
                html_view = f"""
                <div id="molviewer" style="width: 100%; height: 500px; border: 1px solid #ddd; border-radius: 8px;"></div>
                <script src="https://3Dmol.csb.pitt.edu/build/3Dmol-min.js"></script>
                <script>
                    let element = document.getElementById("molviewer");
                    let config = {{ backgroundColor: 'white' }};
                    let viewer = $3Dmol.createViewer(element, config);
                    let pdb = `{block}`;
                    viewer.addModel(pdb, "mol");
                    
                    // Default Style
                    viewer.setStyle({{}}, {{stick: {{radius: 0.2}}}});
                    
                    // Apply Atom Colors if provided
                    let styleMap = {style_json};
                    if (Object.keys(styleMap).length > 0) {{
                        for (let idx in styleMap) {{
                            viewer.setStyle({{serial: parseInt(idx)}}, {{sphere: {{color: styleMap[idx], radius: 0.5}}, stick: {{color: styleMap[idx], radius: 0.2}} }});
                        }}
                    }}
                    
                    viewer.zoomTo();
                    viewer.render();
                </script>
                """
                components.html(html_view, height=520)
with tab2:
    st.markdown("###  Batch Processing")
    st.markdown("Upload a CSV file containing a column named `smiles` to process multiple molecules at once.")
    
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if 'smiles' in df.columns:
            st.info(f"Loaded {len(df)} molecules.")
            
            if st.button(" Run Batch Prediction"):
                progress_bar = st.progress(0)
                preds = []
                
                for i, row in df.iterrows():
                    g_data = smiles_to_graph(row['smiles'])
                    if g_data:
                        with torch.no_grad():
                            batch = torch.zeros(g_data.x.shape[0], dtype=torch.long)
                            val = model(g_data.x, g_data.edge_index, batch).item()
                            preds.append(val)
                    else:
                        preds.append(None)
                    progress_bar.progress((i + 1) / len(df))
                
                df['predicted_log_solubility'] = preds
                st.success("Processing Complete!")
                st.dataframe(df.head())
                
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("⬇ Download Results", csv, "predictions.csv", "text/csv")
        else:
            st.error("CSV must contain a 'smiles' column!")
with tab3:
    st.markdown("###  MLOps Dashboard")
    st.markdown("This project uses **Weights & Biases** for experiment tracking.")
    st.markdown("The chart below shows the training run that generated the current model.")
    st.info("Since we used MLOps, logs are hosted on the cloud.")
    st.markdown(f"[ View Training Curves on WandB](https://wandb.ai/home)")
    st.image("https://wandb.ai/logo.png", width=100)