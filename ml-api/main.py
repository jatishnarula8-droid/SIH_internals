from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers.health import router as health_router
from routers.pest import router as pest_router

app = FastAPI(title="किसानसारथी")

# Enable CORS for all origins (prototype only)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(pest_router)

@app.get("/")
def read_root():
    return {"status": "ok", "service": "crop-health-ai"}
