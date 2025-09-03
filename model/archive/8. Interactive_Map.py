import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# List of calibration analytes
calibration_set = {
    'Benzenesulfonate',
    'Ethanesulfonate',
    'n-Butyrate',
    'Tartrate',
    'Citraconate',
    'Mesaconate',
    'Chromate',
    'Trifluoroacetate',
    'Fluoroacetate',
    'Bromoacetate',
    'Benzoate',
    'Malonate',
    'Pyruvate'
}

# Load your dataset
df = pd.read_csv(r"C:\Users\david.moore\OneDrive - Thermo Fisher Scientific\Desktop\Cluster\MordredVersion\.venv\pca_space_95pct.csv")
ion_types = df['IonType'].dropna().unique()
selected_types = st.multiselect("Filter by Ion Type", options=ion_types, default=ion_types)
pc_columns = [col for col in df.columns if col.startswith("PC")]

# Add a 'Calibration' column
df["Calibration"] = df["Name"].isin(calibration_set)

x_axis = st.sidebar.selectbox("X-axis", df.columns[0:52], index=0)
y_axis = st.sidebar.selectbox("Y-axis", df.columns[0:52], index=1)
z_axis = st.sidebar.selectbox("Z-axis", df.columns[0:52], index=2)

filtered_df = df[df['IonType'].isin(selected_types)]
fig = px.scatter_3d(filtered_df, x=x_axis, y=y_axis, z=z_axis, color='Calibration', color_discrete_map={True: 'orange', False: 'gray'}, hover_data=['Name', 'SMILES'],title='PCA Chemical Similarity Map')

st.title("Chemical Similarity Map")
st.plotly_chart(fig, use_container_width=True, height=700)

# Download matrix (optional future step)
st.download_button("Download CSV", df.to_csv(index=False), file_name="analytes.csv")

pc_df = df[pc_columns].copy()
similarity_matrix = cosine_similarity(pc_df)
sim_df = pd.DataFrame(similarity_matrix, index=df['Name'], columns=df['Name'])

st.download_button(
    label="Download Similarity Matrix (CSV)",
    data=sim_df.to_csv(),
    file_name="similarity_matrix.csv",
    mime='text/csv'
)