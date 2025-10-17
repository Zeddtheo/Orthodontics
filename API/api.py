# API/api.py
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from utils.pano_core import core

# 与约定一致：若未显式指定则默认使用 GPU0
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

app = FastAPI(
    title="MeshSegNet→PointNetReg 推理 API",
    description="输入两个 STL 路径，内部跑分割与 landmark 标注，输出上/下颌 JSON",
    version="1.1",
)


class PairPaths(BaseModel):
    path_a: str = Field(..., description="第一个 STL 的绝对路径")
    path_b: str = Field(..., description="第二个 STL 的绝对路径")
    include_intermediate: bool = Field(False, description="是否在结果里附上中间产物路径")


@app.post("/infer")
def infer(data: PairPaths):
    """STL -> MeshSegNet -> PointNetReg -> 上/下颌 landmark JSON."""
    try:
        output = core(
            path_a=data.path_a,
            path_b=data.path_b,
            include_intermediate=data.include_intermediate,
        )
        response = {"status": "success", "result": output["result"]}
        if "artifacts" in output:
            response["artifacts"] = output["artifacts"]
        return response
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc


__all__ = ["app"]
