from fastapi import APIRouter, Query
from pydantic import BaseModel
from typing import List
from datetime import datetime

# Import the engine (this will work as long as we run uvicorn from the ml-api dir)
from inference.ndvi_engine import get_field_ndvi

router = APIRouter(prefix="/fields", tags=["Field Health"])

class ZoneData(BaseModel):
    zone_id: int
    center_lat: float
    center_lon: float
    ndvi_value: float
    health_status: str

class FieldHealthResponse(BaseModel):
    field_id: str
    overall_health_score: float
    overall_status: str
    zones: List[ZoneData]
    last_updated: str

@router.get("/{field_id}/health", response_model=FieldHealthResponse)
def get_field_health(
    field_id: str,
    lat: float = Query(30.900965, description="Latitude of the field center"),
    lon: float = Query(75.857276, description="Longitude of the field center")
):
    # Fetch zone data from the engine
    zones_data = get_field_ndvi(lat=lat, lon=lon)
    
    # Calculate overall health score
    if zones_data:
        avg_ndvi = sum(zone["ndvi_value"] for zone in zones_data) / len(zones_data)
        avg_ndvi = round(avg_ndvi, 3)
    else:
        avg_ndvi = 0.0
        
    # Determine overall status
    if avg_ndvi > 0.6:
        overall_status = "healthy"
    elif avg_ndvi >= 0.3:
        overall_status = "moderate"
    else:
        overall_status = "stressed"
        
    return FieldHealthResponse(
        field_id=field_id,
        overall_health_score=avg_ndvi,
        overall_status=overall_status,
        zones=zones_data,
        last_updated=datetime.utcnow().isoformat() + "Z"
    )
