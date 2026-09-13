import requests
import pandas as pd

## Feting Topology data of the given Latitude and Longitude
def get_topology(lat: float, lon: float, target_date: str) -> dict:
    """
    Get elevation for a given latitude and longitude.
    
    Args:
        lat: Latitude
        lon: Longitude
        
    Returns:
        Elevation in meters
    """
    target_date = str(target_date).replace("/", "-")
    target_dt = pd.to_datetime(target_date).tz_localize("UTC")
    start_dt = target_dt - pd.Timedelta(days=7)

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_dt.strftime("%Y-%m-%d"),
            "end_date": target_dt.strftime("%Y-%m-%d"),
            "hourly": "soil_moisture_0_to_7cm,soil_moisture_7_to_28cm,precipitation",
            "timezone": "UTC",
        }
    
    try: 
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if not data or "hourly" not in data:
            return {
                "soil_moisture_0_7cm_avg": 0.0,
                "soil_moisture_7_28cm_avg": 0.0,
                "daily_rainfall_mm": 0.0,
                "cumulative_rainfall_3d_mm": 0.0,
                "cumulative_rainfall_7d_mm": 0.0,
            }

        hourly = data["hourly"]
        df = pd.DataFrame(
            {
                "time": pd.to_datetime(hourly["time"], utc=True),
                "soil_0_7": hourly["soil_moisture_0_to_7cm"],
                "soil_7_28": hourly["soil_moisture_7_to_28cm"],
                "precip": hourly["precipitation"],
            }
        )

        target_mask = df["time"].dt.date == target_dt.date()
        past_3d_mask = df["time"] >= (target_dt - pd.Timedelta(days=3))

        '''
        print(df) 
        ## Will print the selected day's every hour rainfall and soil moisture values
        print(df.loc[target_mask, "soil_0_7"])
        print(df.loc[target_mask, "soil_7_28"])
        print(df.loc[target_mask, "precip"])
        '''

        return {
            "soil_moisture_0_7cm_avg": df.loc[target_mask, "soil_0_7"].mean(),
            "soil_moisture_7_28cm_avg": df.loc[target_mask, "soil_7_28"].mean(),
            "daily_rainfall_mm": df.loc[target_mask, "precip"].sum(),
            "cumulative_rainfall_3d_mm": df.loc[past_3d_mask, "precip"].sum(),
            "cumulative_rainfall_7d_mm": df["precip"].sum(),
        }

    except Exception as e:
        print(e)
        return {
            "soil_moisture_0_7cm_avg": 0.0,
            "soil_moisture_7_28cm_avg": 0.0,
            "daily_rainfall_mm": 0.0,
            "cumulative_rainfall_3d_mm": 0.0,
            "cumulative_rainfall_7d_mm": 0.0,
        }

if __name__ == "__main__":
    lat = 28.6139
    lon = 77.2090
    topology = get_topology(lat, lon, "2023/7/17")
    print(f"Topology for {lat}, {lon}: {topology}")