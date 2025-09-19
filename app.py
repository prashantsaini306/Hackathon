import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Weather Vectors: From Scalars to Directional Insights")
st.write("Upload a CSV of gridded temperature data to compute and visualize directional vectors.")

uploaded_file = st.file_uploader("Upload temperature CSV", type=["csv"])

if uploaded_file:
    import pandas as pd
    df = pd.read_csv(uploaded_file)

    grid = df.pivot(index='Lat', columns='Lon', values='Temp').sort_index(ascending=False)
    lat_vals = grid.index.values
    lon_vals = grid.columns.values
    temp_2d = grid.values

    dT_dlat, dT_dlon = np.gradient(temp_2d)

    fig, ax = plt.subplots(figsize=(8,6))
    c = ax.pcolormesh(lon_vals, lat_vals, temp_2d, shading='auto', cmap='coolwarm')
    fig.colorbar(c, ax=ax, label='Temperature (°C)')
    ax.quiver(lon_vals, lat_vals, dT_dlon, dT_dlat, color='black', scale=50)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Temperature Directional Vectors')
    st.pyplot(fig)

    mean_dx = np.nanmean(dT_dlon)
    mean_dy = np.nanmean(dT_dlat)
    direction_deg = np.degrees(np.arctan2(mean_dy, mean_dx))
    magnitude = np.sqrt(mean_dx**2 + mean_dy**2)
    st.write(f"**Average flow direction:** {direction_deg:.1f}°")
    st.write(f"**Average flow magnitude:** {magnitude:.3f}")
else:
    st.info("Upload a CSV with columns: `Lat`, `Lon`, `Temp`.")
