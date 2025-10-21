from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator

from utils.pano_core import core

REPO_DIR = Path(__file__).resolve().parent
VENDOR_DIR = REPO_DIR / "vendor"
if str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))


app = FastAPI(
    title="Orthodontic Landmark API",
    version="1.0.0",
    description="Run ios-model + PointNet-Reg pipeline and return landmark coordinates.",
)


class PipelineRequest(BaseModel):
    upper_path: Path = Field(..., description="Absolute path to the upper jaw STL (or JSON) file.")
    lower_path: Path = Field(..., description="Absolute path to the lower jaw STL (or JSON) file.")
    keep_intermediate: bool = Field(False, description="Keep intermediate artifacts on disk.")

    @validator("upper_path", "lower_path")
    def _path_must_exist(cls, value: Path) -> Path:
        if not value.exists():
            raise ValueError(f"Input path not found: {value}")
        return value


class PipelineResponse(BaseModel):
    result: Dict[str, Any]
    artifacts: Optional[Dict[str, Any]] = None


@app.get("/healthz", tags=["system"])
def health_check() -> Dict[str, str]:
    """Simple readiness probe."""
    return {"status": "ok"}


@app.post("/pipeline", response_model=PipelineResponse, tags=["pipeline"])
def run_pipeline(request: PipelineRequest) -> Dict[str, Any]:
    """
    Execute the segmentation + landmark pipeline.
    """
    try:
        payload = core(
            path_a=str(request.upper_path),
            path_b=str(request.lower_path),
            include_intermediate=request.keep_intermediate,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return dict(payload)
