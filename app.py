import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
from sklearn.cluster import KMeans

st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("Customer Segmentation EDA and Clustering")

# --- File Upload and Defaults ---
st.sidebar.header("Upload your dataset (optional)")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    # Default data
    df = pd.read_csv("Mall_Customers.csv")  # Ensure this file is in your repo/app folder

# --- Basic EDA ---
st.subheader("Dataset Preview")
st.write(df.head())

st.subheader("Summary Statistics")
st.write(df.describe())

st.subheader("Gender Distribution")
st.bar_chart(df['Gender'].value_counts())

# --- Clustering ---
st.subheader("Customer Segmentation Clustering")

# Select columns
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Apply KMeans
kmeans = KMeans(n_clusters=5, random_state=42)
df['Spending and Income Cluster'] = kmeans.fit_predict(X)

# Add centers
centers = pd.DataFrame(kmeans.cluster_centers_, columns=['x', 'y'])

# --- Plot Clusters ---
fig, ax = plt.subplots(figsize=(10, 8))
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', 
                hue='Spending and Income Cluster', palette='tab10', ax=ax)
ax.scatter(x=centers['x'], y=centers['y'], s=100, c='black', marker='*')
ax.set_title("Customer Segmentation Clusters")

st.pyplot(fig)

# --- Download Buttons ---
# Image Download
img_buffer = io.BytesIO()
fig.savefig(img_buffer, format='png')
img_buffer.seek(0)
st.download_button("Download Cluster Plot", data=img_buffer, file_name="clustering.png", mime="image/png")

# CSV Download
csv_buffer = io.StringIO()
df.to_csv(csv_buffer, index=False)
csv_data = csv_buffer.getvalue()
st.download_button("Download Clustering CSV", data=csv_data, file_name="Clustering.csv", mime="text/csv")
