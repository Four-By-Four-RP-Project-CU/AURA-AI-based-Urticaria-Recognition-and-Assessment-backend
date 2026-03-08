"""
Root-level FastAPI application for AURA backend integration.
Mounts individual student modules as sub-applications.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Import individual student apps
from IT22577160.app.main import app as it22577160_app
from IT22607232.app.Risk_main import app as it22607232_app

# Create root application
app = FastAPI(
    title="AURA - AI-based Urticaria Recognition and Assessment",
    description="Multi-module backend API for CSU diagnosis and treatment recommendation",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root health check
@app.get("/")
def root():
    return {
        "message": "AURA Backend API",
        "version": "1.0.0",
        "modules": {
            "IT22577160": "/IT22577160",
            "IT22607232": "/IT22607232",
        }
    }

@app.get("/health")
def health():
    return {"status": "ok", "service": "AURA Root API"}

# Mount student modules
# IT22577160 - CSU Decision Support (Lab extraction, Prediction, PDF reports)
app.mount("/IT22577160", it22577160_app)

# IT22607232 - CU Risk Profiling (Multi-task classification, OCR lab reports)
app.mount("/IT22607232", it22607232_app)

# Add more student modules here:
# Example:
# from IT22XXXXXX.app.main import app as other_app
# app.mount("/IT22XXXXXX", other_app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
