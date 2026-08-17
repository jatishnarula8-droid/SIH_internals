# MOCK DATA — replace with real Google Earth Engine Sentinel-2 integration once GEE project registration is confirmed

import json
import random
import math
from typing import List, Dict, Any

def get_field_ndvi(lat: float, lon: float, buffer_meters: int = 500, grid_size: int = 5) -> List[Dict[str, Any]]:
    """
    Simulates NDVI field data for a specified region without calling any external APIs.
    Generates realistic, clustered zones (stressed, moderate, healthy).
    """
    # 1 degree of latitude is approx 111,320 meters
    # For longitude, it depends on latitude, but we'll use a simple approximation for small buffers
    lat_offset = buffer_meters / 111320.0
    lon_offset = buffer_meters / (111320.0 * math.cos(math.radians(lat)))
    
    min_lon = lon - lon_offset
    max_lon = lon + lon_offset
    min_lat = lat - lat_offset
    max_lat = lat + lat_offset
    
    step_lon = (max_lon - min_lon) / grid_size
    step_lat = (max_lat - min_lat) / grid_size
    
    # Seed the random generator using the base coordinates so the output is reproducible
    random.seed(f"{lat:.4f}_{lon:.4f}")
    
    # Pick a random center for our "stressed" disease/water-stress patch
    # We constrain it slightly so it doesn't always touch the extreme edges
    stress_center_i = random.randint(1, max(1, grid_size - 2))
    stress_center_j = random.randint(1, max(1, grid_size - 2))
    
    output_zones = []
    zone_id_counter = 1
    
    for i in range(grid_size):
        for j in range(grid_size):
            # Calculate coordinates for this specific cell
            lon1 = min_lon + i * step_lon
            lat1 = min_lat + j * step_lat
            lon2 = lon1 + step_lon
            lat2 = lat1 + step_lat
            
            center_lon = (lon1 + lon2) / 2.0
            center_lat = (lat1 + lat2) / 2.0
            
            # Calculate distance to the "stressed" center (Euclidean distance on the grid)
            dist = math.sqrt((i - stress_center_i)**2 + (j - stress_center_j)**2)
            
            # Re-seed using the specific zone coordinates for small variances within the zone
            random.seed(f"{center_lat:.6f}_{center_lon:.6f}")
            
            if dist <= 1.0:
                # The core stressed patch (e.g. distance 0 or 1, gives ~1-3 patches)
                ndvi_val = random.uniform(0.15, 0.29)
            elif dist <= 2.0:
                # Transition zones / moderate stress around the core patch
                ndvi_val = random.uniform(0.3, 0.59)
            else:
                # Healthy crop zones making up the rest of the field
                ndvi_val = random.uniform(0.61, 0.85)
            
            # Determine health status based on NDVI value
            if ndvi_val > 0.6:
                health = "healthy"
            elif ndvi_val >= 0.3:
                health = "moderate"
            else:
                health = "stressed"
                
            output_zones.append({
                "zone_id": zone_id_counter,
                "center_lat": round(center_lat, 6),
                "center_lon": round(center_lon, 6),
                "ndvi_value": round(ndvi_val, 3),
                "health_status": health
            })
            zone_id_counter += 1
            
    return output_zones

if __name__ == "__main__":
    # Sample coordinate: Agricultural area near Ludhiana, Punjab
    sample_lat = 30.900965
    sample_lon = 75.857276
    
    print(f"Fetching (MOCK) NDVI for grid centered at (lat: {sample_lat}, lon: {sample_lon})...")
    try:
        zones_data = get_field_ndvi(sample_lat, sample_lon)
        print("\n--- Field NDVI Data ---")
        print(json.dumps(zones_data, indent=2))
    except Exception as e:
        print(f"Error occurred: {e}")
