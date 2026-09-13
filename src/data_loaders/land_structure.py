import numpy as np
import pandas as pd
import rasterio
import os
from rasterio.windows import Window

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "data", "land_structure", "rasterexportedLayerNE.tif"))

import numpy as np
import rasterio
from rasterio.windows import Window

def get_terrain_features(lat: float, lon: float, dem_path: str = DATA_DIR) -> dict:
    '''
        The terrain features are extracted from the Digital Elevation Model (DEM).
    '''
    with rasterio.open(dem_path) as src:
        row, col = src.index(lon, lat)
        
        # Read 3x3 pixel window
        win = Window(col - 1, row - 1, 3, 3)
        data = src.read(1, window=win).astype(float)
        
        if data.shape != (3, 3):
            raise ValueError("Coordinate is out of bounds.")
        
        z0 = data[1, 1]
        
        # Accurate meter conversion per degree at this specific latitude
        scale_x = 111320 * np.cos(np.radians(lat))
        scale_y = 110540
        
        dx = abs(src.res[0]) * scale_x
        dy = abs(src.res[1]) * scale_y
        
        # 3x3 neighborhood
        z1, z2, z3 = data[0, 0], data[0, 1], data[0, 2]
        z4, _,  z5 = data[1, 0], data[1, 1], data[1, 2]
        z6, z7, z8 = data[2, 0], data[2, 1], data[2, 2]
        
        # Horn's Slope
        dz_dx = ((z3 + 2 * z5 + z8) - (z1 + 2 * z4 + z6)) / (8 * dx)
        dz_dy = ((z6 + 2 * z7 + z8) - (z1 + 2 * z2 + z3)) / (8 * dy)
        slope = float(np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))))
        
        return {
            'elevation': round(float(z0), 2),
            'slope': round(slope, 4),
            'roughness': round(float(np.max(data) - np.min(data)), 4),
            'tri': round(float(np.sqrt(np.sum((data - z0) ** 2))), 4)
        }

if __name__ == "__main__":
    user_lat = 23.228159
    user_lon = 91.494286
    
    # Set your exact scale ratio (10860 or 10600)
    results = get_terrain_features(user_lat, user_lon, dem_path=DATA_DIR)
    
    # Pass straight to your ML model DataFrame
    df_for_model = pd.DataFrame([results])
    print(df_for_model)