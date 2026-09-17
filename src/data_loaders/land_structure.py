import numpy as np
import pandas as pd
import requests

# ~30-meter resolution (1 arc-second) matching standard SRTM/Copernicus DEM grids
DEM_STEP_DEG = 0.000278


def get_terrain_features(
    lat: float, lon: float, step: float = DEM_STEP_DEG
) -> dict:
    """Extracts elevation, Horn's slope, roughness, and TRI via Open-Meteo API.

    Replaces local DEM raster files with a single 9-point batch query.
    """
    # 3x3 grid around (lat, lon): Row 0 (North), Row 1 (Center), Row 2 (South)
    lats = [
        lat + step,
        lat + step,
        lat + step,  # Row 0
        lat,
        lat,
        lat,  # Row 1
        lat - step,
        lat - step,
        lat - step,  # Row 2
    ]
    lons = [
        lon - step,
        lon,
        lon + step,  # Row 0
        lon - step,
        lon,
        lon + step,  # Row 1
        lon - step,
        lon,
        lon + step,  # Row 2
    ]

    url = "https://api.open-meteo.com/v1/elevation"
    params = {
        "latitude": ",".join(map(str, lats)),
        "longitude": ",".join(map(str, lons)),
    }

    try:
        res = requests.get(url, params=params, timeout=5)
        res.raise_for_status()
        elevations = res.json().get("elevation", [])

        if len(elevations) != 9:
            raise ValueError("Incomplete elevation response received.")

        # Reshape into identical 3x3 window
        data = np.array(elevations, dtype=float).reshape((3, 3))
        z0 = data[1, 1]

        # Metric ground distance per degree at this latitude
        dx = step * (111320 * np.cos(np.radians(lat)))
        dy = step * 110540

        # Neighborhood extraction (same indexing as local raster)
        z1, z2, z3 = data[0, 0], data[0, 1], data[0, 2]
        z4, _, z5 = data[1, 0], data[1, 1], data[1, 2]
        z6, z7, z8 = data[2, 0], data[2, 1], data[2, 2]

        # Horn's Slope
        dz_dx = ((z3 + 2 * z5 + z8) - (z1 + 2 * z4 + z6)) / (8 * dx)
        dz_dy = ((z6 + 2 * z7 + z8) - (z1 + 2 * z2 + z3)) / (8 * dy)
        slope = float(np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))))

        return {
            "elevation": round(float(z0), 2),
            "slope": round(slope, 4),
            "roughness": round(float(np.max(data) - np.min(data)), 4),
            "tri": round(float(np.sqrt(np.sum((data - z0) ** 2))), 4),
        }

    except Exception as e:
        # Fallback safeguard for live demos if Wi-Fi drops
        print(f"Warning: Terrain API query failed ({e}) — using baseline fallback.")
        return {
            "elevation": 420.0,
            "slope": 18.5000,
            "roughness": 22.0000,
            "tri": 28.5000,
        }


if __name__ == "__main__":
    user_lat = 23.228159
    user_lon = 91.494286

    results = get_terrain_features(user_lat, user_lon)
    df_for_model = pd.DataFrame([results])
    print(df_for_model)