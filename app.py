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
uploaded_file = st.file_uploader("Please upload your dataset with one of the above parameter it contains :", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # --- build datetime ---
    df = df.rename(columns={'Year':'year', 'Month':'month', 'Date':'day'})
    df["datetime"] = pd.to_datetime(
        df['year'].astype(str) + '-' + df['month'].astype(str) + '-' + df['day'].astype(str)
    )

    # ================= TEMPERATURE =================
    if parameter == "Temperature":
        import re
        # --- Extract T2M_<lat>_<lon> columns ---
        t2m_columns = [c for c in df.columns if re.match(r"WS10M_\d+_\d+$", c)]

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
        ax.set_title(f"Temperature 🌡️ directional vectors on {sel_date}")
        st.pyplot(fig)

        # Flow info
        mean_dx = np.nanmean(dT_dlon)
        mean_dy = np.nanmean(dT_dlat)
        direction_deg = np.degrees(np.arctan2(mean_dy, mean_dx))
        mean_magnitude = np.nanmean(np.sqrt(dT_dlon**2 + dT_dlat**2))

        st.write(f"**Dominant flow direction:** {direction_deg:.1f}°")
        st.write(f"**Average flow magnitude:** {mean_magnitude:.2f}")

        # ==========================================================
        # 🔎 Extra ANALYSIS SECTION
        # ==========================================================

        # --- 1. Statistical summary ---
        temp_values = temp_2d.flatten()
        temp_values = temp_values[~np.isnan(temp_values)]
        summary = {
            "Mean (°C)": np.mean(temp_values),
            "Median (°C)": np.median(temp_values),
            "Min (°C)": np.min(temp_values),
            "Max (°C)": np.max(temp_values),
            "Std Dev (°C)": np.std(temp_values),
        }
        st.write("### 📊 Statistical Summary")
        st.table(pd.DataFrame(summary, index=["Statistics"]))

        # --- 2. Distribution Histogram ---
        fig, ax = plt.subplots()
        ax.hist(temp_values, bins=20, color="tomato", edgecolor="black", alpha=0.7)
        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("Frequency")
        ax.set_title("Temperature Distribution")
        st.pyplot(fig)

        # --- 3. Time-series Trend of Mean Temperature ---
        mean_temp_series = np.nanmean(data_cube, axis=(1,2))  # avg over lat/lon
        time_vals = df["datetime"].unique()

        fig, ax = plt.subplots()
        ax.plot(time_vals, mean_temp_series, color="darkred")
        ax.set_xlabel("Date")
        ax.set_ylabel("Mean Temperature (°C)")
        ax.set_title("Mean Temperature Trend Over Time")
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # --- 4. Monthly Cycle ---
        df["month"] = df["datetime"].dt.month
        monthly_mean = df.groupby("month")[t2m_columns].mean().mean(axis=1)

        fig, ax = plt.subplots()
        monthly_mean.plot(kind="bar", ax=ax, color="salmon")
        ax.set_xlabel("Month")
        ax.set_ylabel("Avg Temperature (°C)")
        ax.set_title("Monthly Temperature Cycle")
        st.pyplot(fig)

    # ================= RAINFALL =================
    elif parameter == "Rainfall":
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
        ax.set_title(f"Rainfall 🌧️ directional vectors on {sel_date}")
        st.pyplot(fig)

        # --- Flow info ---
        mean_dx = np.nanmean(dP_dlon)
        mean_dy = np.nanmean(dP_dlat)
        direction_deg = np.degrees(np.arctan2(mean_dy, mean_dx))
        mean_magnitude = np.nanmean(np.sqrt(dP_dlon**2 + dP_dlat**2))

        st.write(f"**Dominant flow direction:** {direction_deg:.1f}°")
        st.write(f"**Average flow magnitude:** {mean_magnitude:.2f}")
        

    # ================= WIND SPEED =================
    elif parameter == "Wind Speed":
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

        # --- Date selector ---
        sel_date = st.selectbox("Select a date", df["datetime"].dt.strftime("%Y-%m-%d").unique())
        t_idx = df.index[df["datetime"] == sel_date][0]
        wind_2d = data_cube[t_idx, :, :]

        lon_2d, lat_2d = np.meshgrid(lon_vals, lat_vals)

        # --- Gradient (to show directional changes in speed) ---
        dW_dlat, dW_dlon = np.gradient(wind_2d, lat_vals, lon_vals)

        # --- Plot wind speed map + gradient arrows ---
        fig, ax = plt.subplots(figsize=(8,6))
        pcm = ax.pcolormesh(lon_2d, lat_2d, wind_2d, shading='auto', cmap='viridis')
        fig.colorbar(pcm, ax=ax, label='Wind Speed (m/s)')
        ax.quiver(lon_2d, lat_2d, dW_dlon, dW_dlat, scale=50, color='black')
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"Wind speed 🌬️ directional vectors on {sel_date}")
        st.pyplot(fig)

        # --- Flow info ---
        mean_dx = np.nanmean(dW_dlon)
        mean_dy = np.nanmean(dW_dlat)
        direction_deg = np.degrees(np.arctan2(mean_dy, mean_dx))
        mean_magnitude = np.nanmean(np.sqrt(dW_dlon**2 + dW_dlat**2))

        st.write(f"**Dominant flow direction:** {direction_deg:.1f}°")
        st.write(f"**Average flow magnitude:** {mean_magnitude:.2f}")

    # ================= POLLUTANTS =================
    elif parameter == "Pollutants":
        st.info("Pollutant data visualization will appear here.")
