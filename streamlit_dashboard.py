import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(page_title="Drug Discovery GNN Dashboard", layout="centered")
st.title("📊 GNN Training Performance Dashboard")
st.write("Visualizing the model loss across training epochs.")

# Load training history
df = pd.read_csv("train_history.csv")

# Plot loss curve
chart = alt.Chart(df).mark_line(point=True).encode(
    x='epoch',
    y='loss',
    tooltip=['epoch', 'loss']
).properties(
    title='Training Loss Curve'
)

st.altair_chart(chart, use_container_width=True)
