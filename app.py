import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

st.title("Weather Gradient Tracker")

# ---- Dropdown menu ----
parameter = st.selectbox(
    "Select a parameter to analyze:",
    ["Temperature 🌡️", "Rainfall 🌧️", "Wind Speed 🌬️", "Pollutants 💨"]
)

# ---- File Upload ----
uploaded_file = st.file_uploader("Upload your CSV file here:", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # ================= TEMPERATURE =================
    if parameter == "Temperature 🌡️":

        # --- build datetime ---
        df = df.rename(columns={'Year':'year', 'Month':'month', 'Date':'day'})
        df["datetime"] = pd.to_datetime(
            df['year'].astype(str) + '-' + df['month'].astype(str) + '-' + df['day'].astype(str)
        )
        # --- Extract T2M_<lat>_<lon> columns ---
        t2m_columns = [c for c in df.columns if re.match(r'^T2M_\d+_\d+$', c)]

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

        # Plot
        fig, ax = plt.subplots(figsize=(8,6))
        pcm = ax.pcolormesh(lon_2d, lat_2d, temp_2d, shading='auto', cmap='coolwarm')
        fig.colorbar(pcm, ax=ax, label='Temperature (°C)')
        ax.quiver(lon_2d, lat_2d, dT_dlon, dT_dlat, scale=50, color='black')
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"Temperature directional vectors on {sel_date}")
        st.pyplot(fig)

        # Flow info
        mean_dx = np.nanmean(dT_dlon)
        mean_dy = np.nanmean(dT_dlat)
        direction_deg = np.degrees(np.arctan2(mean_dy, mean_dx))
        mean_magnitude = np.nanmean(np.sqrt(dT_dlon**2 + dT_dlat**2))

        st.write(f"**Dominant flow direction:** {direction_deg:.1f}°")
        st.write(f"**Average flow magnitude:** {mean_magnitude:.2f}")
        
    # ================= RAINFALL =================
    elif parameter == "Rainfall 🌧️":
        # --- build datetime ---
        df = df.rename(columns={'Year':'year', 'Month':'month', 'Date':'day'})
        df["datetime"] = pd.to_datetime(
            df['year'].astype(str) + '-' + df['month'].astype(str) + '-' + df['day'].astype(str)
        )
        # --- Extract Prec_<lat>_<lon> columns ---
        prec_columns = [c for c in df.columns if re.match(r"Prec_\d+_\d+$", c)]

        # --- Get unique lat/lon values ---
        lat_vals = sorted({int(c.split("_")[1])/100 if len(c.split("_")[1]) > 2 else int(c.split("_")[1])
                           for c in prec_columns})
        lon_vals = sorted({int(c.split("_")[2])/100 if len(c.split("_")[2]) > 2 else int(c.split("_")[2])
                           for c in prec_columns})

        ntime = len(df)
        nlat, nlon = len(lat_vals), len(lon_vals)
        data_cube = np.full((ntime, nlat, nlon), np.nan)

        # --- Fill cube ---
        for c in prec_columns:
            lat_raw, lon_raw = c.split("_")[1:]
            lat_val = float(lat_raw)/100 if len(lat_raw) > 2 else float(lat_raw)
            lon_val = float(lon_raw)/100 if len(lon_raw) > 2 else float(lon_raw)
            i = lat_vals.index(lat_val)
            j = lon_vals.index(lon_val)
            data_cube[:, i, j] = df[c].values

        # --- Date selector ---
        sel_date = st.selectbox("Select a date", df["datetime"].dt.strftime("%Y-%m-%d").unique())
        t_idx = df.index[df["datetime"] == sel_date][0]
        prec_2d = data_cube[t_idx, :, :]

        lon_2d, lat_2d = np.meshgrid(lon_vals, lat_vals)

        # --- Gradient ---
        dP_dlat, dP_dlon = np.gradient(prec_2d, lat_vals, lon_vals)

        # --- Plot rainfall map + directional vectors ---
        fig, ax = plt.subplots(figsize=(8,6))
        pcm = ax.pcolormesh(lon_2d, lat_2d, prec_2d, shading='auto', cmap='Blues')
        fig.colorbar(pcm, ax=ax, label='Rainfall (mm)')
        ax.quiver(lon_2d, lat_2d, dP_dlon, dP_dlat, scale=50, color='black')
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"Rainfall directional vectors on {sel_date}")
        st.pyplot(fig)

        # --- Flow info ---
        mean_dx = np.nanmean(dP_dlon)
        mean_dy = np.nanmean(dP_dlat)
        direction_deg = np.degrees(np.arctan2(mean_dy, mean_dx))
        mean_magnitude = np.nanmean(np.sqrt(dP_dlon**2 + dP_dlat**2))

        st.write(f"**Dominant flow direction:** {direction_deg:.1f}°")
        st.write(f"**Average flow magnitude:** {mean_magnitude:.2f}")
        

    # ================= WIND SPEED =================
    elif parameter == "Wind Speed 🌬️":
        # --- build datetime ---
        df = df.rename(columns={'Year':'year', 'Month':'month', 'Date':'day'})
        df["datetime"] = pd.to_datetime(
            df['year'].astype(str) + '-' + df['month'].astype(str) + '-' + df['day'].astype(str)
        )
        # --- Extract WS10M_<lat>_<lon> columns ---
        wind_columns = [c for c in df.columns if re.match(r"WS10M_\d+_\d+$", c)]

        # --- Get unique lat/lon values ---
        lat_vals = sorted({int(c.split("_")[1])/100 if len(c.split("_")[1]) > 2 else int(c.split("_")[1])
                           for c in wind_columns})
        lon_vals = sorted({int(c.split("_")[2])/100 if len(c.split("_")[2]) > 2 else int(c.split("_")[2])
                           for c in wind_columns})

        ntime = len(df)
        nlat, nlon = len(lat_vals), len(lon_vals)
        data_cube = np.full((ntime, nlat, nlon), np.nan)

        # --- Fill cube ---
        for c in wind_columns:
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

        # --- Gradient (to show directional changes in speed) ---
        dW_dlat, dW_dlon = np.gradient(temp_2d, lat_vals, lon_vals)

        # --- Plot wind speed map + gradient arrows ---
        fig, ax = plt.subplots(figsize=(8,6))
        pcm = ax.pcolormesh(lon_2d, lat_2d, temp_2d, shading='auto', cmap='viridis')
        fig.colorbar(pcm, ax=ax, label='Wind Speed (m/s)')
        ax.quiver(lon_2d, lat_2d, dW_dlon, dW_dlat, scale=50, color='black')
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"Wind speed directional vectors on {sel_date}")
        st.pyplot(fig)

        # --- Flow info ---
        mean_dx = np.nanmean(dW_dlon)
        mean_dy = np.nanmean(dW_dlat)
        direction_deg = np.degrees(np.arctan2(mean_dy, mean_dx))
        mean_magnitude = np.nanmean(np.sqrt(dW_dlon**2 + dW_dlat**2))

        st.write(f"**Dominant flow direction:** {direction_deg:.1f}°")
        st.write(f"**Average flow magnitude:** {mean_magnitude:.2f}")


    elif parameter == "Pollutants 💨":

       # streamlit_app.py
        import streamlit as st
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        # ----------------------------
        # 1. Load your CSV and build cube
        # ----------------------------
        @st.cache_data
        def load_cube(csv_path):
            df = pd.read_csv(csv_path)

            # --- Build data cube ---
            lat_vals = sorted(df['lat'].unique())
            lon_vals = sorted(df['lon'].unique())
            time_vals = pd.to_datetime(df.columns[2:], format="%Y-%m")  # adapt if needed
        
            ntime, nlat, nlon = len(time_vals), len(lat_vals), len(lon_vals)
            data_cube = np.full((ntime, nlat, nlon), np.nan)
        
            lat_index = {lat: i for i, lat in enumerate(lat_vals)}
            lon_index = {lon: j for j, lon in enumerate(lon_vals)}
        
            for _, row in df.iterrows():
                i = lat_index[row['lat']]
                j = lon_index[row['lon']]
                data_cube[:, i, j] = row.values[2:]

            return data_cube, time_vals, lat_vals, lon_vals

        # ----------------------------
        # 2. Gradient plotting function
        # ----------------------------
        def plot_pmi_gradient(data_cube, time_vals, lat_vals, lon_vals, t_index,
                              extent=[72,82,28,38], cmap="coolwarm"):
            field = data_cube[t_index, :, :]
            lon_2d, lat_2d = np.meshgrid(lon_vals, lat_vals, indexing="xy")
        
            # Gradients
            dT_dlat, dT_dlon = np.gradient(field, lat_vals, lon_vals)
        
            # Normalize to unit length
            mag = np.sqrt(dT_dlon**2 + dT_dlat**2)
            mag[mag == 0] = 1.0
            u = dT_dlon / mag
            v = dT_dlat / mag
        
            # Rescale to visible arrows
            dx = np.median(np.diff(lon_vals))
            dy = np.median(np.diff(lat_vals))
            arrow_len = 0.5 * min(dx, dy)
            u *= arrow_len
            v *= arrow_len
        
            # Plot
            proj = ccrs.PlateCarree()
            fig, ax = plt.subplots(figsize=(8,6), subplot_kw={'projection': proj})
        
            ax.coastlines(resolution="10m", linewidth=1)
            ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.8)
            ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.5)
        
            pcm = ax.pcolormesh(lon_2d, lat_2d, field,
                                transform=ccrs.PlateCarree(),
                                shading='auto', cmap=cmap)
            plt.colorbar(pcm, ax=ax, label="PMI")
        
            ax.quiver(lon_2d, lat_2d, u, v,
                      transform=ccrs.PlateCarree(),
                      scale=1, color="black", width=0.002)
        
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            ax.set_title(f"PMI Gradient Directions on {time_vals[t_index].strftime('%Y-%m')}", fontsize=12)
            st.pyplot(fig)
        
        # ----------------------------
        # 3. Streamlit UI
        # ----------------------------
        st.title("PMI Gradient Vector Viewer")
        
        # Upload CSV
        csv_file = st.file_uploader("Upload PMI CSV file", type=["csv"])
        if csv_file:
            data_cube, time_vals, lat_vals, lon_vals = load_cube(csv_file)
        
            # Dropdown for time selection
            time_choice = st.selectbox("Select month:", time_vals.strftime("%Y-%m"))
            t_index = np.where(time_vals.strftime("%Y-%m") == time_choice)[0][0]
        
            # Plot
            plot_pmi_gradient(data_cube, time_vals, lat_vals, lon_vals, t_index)

