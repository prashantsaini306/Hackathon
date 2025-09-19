import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import cartopy.crs as ccrs
import cartopy.feature as cfeature

st.title("Weather Gradient Viewer")

# ---- Dropdown menu ----
parameter = st.selectbox(
    "Select a parameter to analyze:",
    ["Temperature", "Rainfall", "Wind Speed", "Pollutants"]
)

# ---- File Upload ----
uploaded_file = st.file_uploader("Upload your CSV (Data1.csv)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # --- build datetime ---
    df = df.rename(columns={'Year':'year', 'Month':'month', 'Date':'day'})
    df["datetime"] = pd.to_datetime(
        df['year'].astype(str) + '-' + df['month'].astype(str) + '-' + df['day'].astype(str)
    )

    # ================= TEMPERATURE =================
    if parameter == "Temperature":
        # --- Extract T2M_<lat>_<lon> columns ---
        t2m_columns = [c for c in df.columns if re.match(r"T2M_\d+_\d+$", c)]

        lat_vals = sorted({int(c.split("_")[1])/100 if len(c.split("_")[1]) > 2 else int(c.split("_")[1])
                           for c in t2m_columns})
        lon_vals = sorted({int(c.split("_")[2])/100 if len(c.split("_")[2]) > 2 else int(c.split("_")[2])
                           for c in t2m_columns})

        ntime = len(df)
        nlat, nlon = len(lat_vals), len(lon_vals)
        data_cube = np.full((ntime, nlat, nlon), np.nan)

        # Fill cube
        for c in t2m_columns:
            lat_raw, lon_raw = c.split("_")[1:]
            lat_val = float(lat_raw)/100 if len(lat_raw) > 2 else float(lat_raw)
            lon_val = float(lon_raw)/100 if len(lon_raw) > 2 else float(lon_raw)
            i = lat_vals.index(lat_val)
            j = lon_vals.index(lon_val)
            data_cube[:, i, j] = df[c].values

        # Date selector
        sel_date = st.selectbox("Select a date", df["datetime"].dt.strftime("%Y-%m-%d").unique())
        t_idx = df.index[df["datetime"] == sel_date][0]
        temp_2d = data_cube[t_idx, :, :]

        lon_2d, lat_2d = np.meshgrid(lon_vals, lat_vals)

        # Gradient
        dT_dlat, dT_dlon = np.gradient(temp_2d, lat_vals, lon_vals)

       # --- Plot with Cartopy ---
proj = ccrs.PlateCarree()
fig = plt.figure(figsize=(10,8))
ax = plt.axes(projection=proj)

ax.coastlines(resolution="10m", linewidth=1)
ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.8)
ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.5)
ax.add_feature(cfeature.OCEAN, facecolor="lightblue", alpha=0.3)

pcm = ax.pcolormesh(lon_2d, lat_2d, temp_2d,
                    transform=ccrs.PlateCarree(),
                    shading='auto', cmap='coolwarm')
plt.colorbar(pcm, ax=ax, orientation="vertical", label="Temperature (°C)")

ax.quiver(lon_2d, lat_2d, dT_dlon, dT_dlat,
          transform=ccrs.PlateCarree(),
          scale=50, color="black")

# 🔹 Restrict to requested region
ax.set_extent([70, 85, 28, 38], crs=ccrs.PlateCarree())
ax.set_title(f"Temperature directional vectors on {sel_date}", fontsize=14)

st.pyplot(fig)

 

    # ================= RAINFALL =================
    elif parameter == "Rainfall":
        st.info("Rainfall visualization will appear here.")

    # ================= WIND SPEED =================
    elif parameter == "Wind Speed":
        st.info("Wind speed visualization will appear here.")

    # ================= POLLUTANTS =================
    elif parameter == "Pollutants":
        st.info("Pollutant data visualization will appear here.")
