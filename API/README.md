# MeshSegNet → PointNet-Reg Quickstart

这份 API 做的事情很简单：给它一对已经分好上下颌的 STL，里面会先跑 MeshSegNet 做分割，再把结果交给 PointNet-Reg 生成牙位 landmark，最后把上颌、下颌的 JSON 一起返回给你。

## 准备工作
- 把 `API/` 整个目录解压到一台带 CUDA 的机器上。
- 模型权重已经放在 `API/models/` 下；如果需要更新自己训练的模型，只要替换同名文件即可。

## 运行方式
### 方式 1：Docker（推荐）
```bash
cd API
docker build -t mesh-workflow:latest .
docker run --rm -it --gpus '"device=0"' -e CUDA_VISIBLE_DEVICES=0 -p 8000:8000 mesh-workflow:latest
```
容器会在 `8000` 端口启动 FastAPI。想用本地数据就把目录挂载进来，比如 `-v /data/cases:/data`。

### 方式 2：本地环境
```bash
cd API
micromamba create -n meshsegnet -f envs/meshsegnet.yml
micromamba create -n pointnetreg -f envs/pointnetreg.yml
micromamba create -n calc -f envs/calc.yml
micromamba run -n calc pip install -r requirements.api.txt
micromamba run -n calc uvicorn api:app --host 0.0.0.0 --port 8000
```

## 调用接口
唯一的入口是 `POST /infer`，请求体需要告诉服务上下颌 STL 的绝对路径：
```bash
curl -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -d '{
        "path_a": "/data/case001_U.stl",
        "path_b": "/data/case001_L.stl",
        "include_intermediate": true
      }'
```

返回示例：
```json
{
  "status": "success",
  "result": {
    "up": {
      "11m": [-8.64, -0.22, 21.40],
      "11ma": [-5.34, -0.22, 21.79],
      "...": "..."
    },
    "low": {
      "31m": [0.60, -0.03, 20.10],
      "31ma": [-1.85, -0.13, 19.54],
      "...": "..."
    }
  },
  "artifacts": {
    "vtp_a": "/app/runs/7b5ad4c8f9c4/A.vtp",
    "vtp_b": "/app/runs/7b5ad4c8f9c4/B.vtp",
    "json_a": "/app/runs/7b5ad4c8f9c4/A.json",
    "json_b": "/app/runs/7b5ad4c8f9c4/B.json"
  }
}
```
把 `include_intermediate` 设为 `false` 时，流程会在返回前自动清理 VTP/JSON 以及整 个工作目录，只保留响应里的 landmark JSON。

## 常见问题
- **STL 路径在哪找？** 把文件挂载进容器或直接用绝对路径，确保 API 进程能读到就行。
- **想要 metrics 吗？** 默认不算。如果需要，只要在代码里调用 `run_pipeline(..., run_metrics=True)`，或自己复用 `wrappers/calc_metrics_wrapper.py`。
- **临时文件在哪里？** 默认会在 `API/runs/` 下建一个随机目录。如果不请求中间产物，接口会在 返回后自动清理该目录。
