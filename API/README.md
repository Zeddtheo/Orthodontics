# ios-model → PointNet-Reg CLI

给定一对分好上下颌的 STL，本工具会先跑 ios-model TorchScript 完成分割，再调用 PointNet-Reg 得到牙位 landmark 坐标，直接输出 JSON 结果。无需再启动 FastAPI 或 Docker。

## Docker 快速运行
```bash
cd API
docker build -t ios-pointnet-cli:latest .
docker run --rm --gpus all \
  -v /abs/data:/data \
  -e UPPER_STL=/data/319_U.stl \
  -e LOWER_STL=/data/319_L.stl \
  -e OUTPUT_JSON=/data/319_result.json \
  ios-pointnet-cli:latest
```
如需保留中间产物，加上 `-e KEEP_INTERMEDIATE=1`；用 `-e PRETTY_JSON=1` 可输出排版好的 JSON。容器中的 `/app` 即 `API/` 目录拷贝，可按需挂载输入或自定义权重。

## 手动准备环境
1. 确保机器安装了 NVIDIA 驱动和 CUDA，对应 GPU 能被 `nvidia-smi` 识别。
2. 在 `API/` 目录中，用 micromamba 创建 3 个环境：
   ```bash
   cd API
   micromamba create -n ios_model -f envs/ios_model.yml
   micromamba create -n pointnetreg -f envs/pointnetreg.yml
   micromamba create -n calc -f envs/calc.yml
   micromamba run -n ios_model pip install --no-cache-dir torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.2+cu121.html
   micromamba run -n ios_model pip install --no-cache-dir torch_geometric==2.5.3
   ```
   如果使用 `release-linux/` 的离线包，先执行 `API/release-linux/bootstrap.sh`，上述环境会自动解压到 `API/.micromamba/` 下。

模型权重都已经内置：PointNet-Reg 在 `API/models/pointnetreg/`，ios-model TorchScript 权重在 `API/models/ios-model/`（仅保留 `.pt`），对应脚本在 `API/vendor/ios-model/`。

### 集成测试（Linux）

```bash
cd API
tests/run_api_tests.sh
```

脚本会自动准备 `calc`、`ios_model`、`pointnetreg` 三个 micromamba 环境（默认安装在 `API/.micromamba/`），随后对 `tests/319_[UL].stl` 运行整套管线，并校验输出 JSON 是否包含完整的牙位坐标。如果只想快速检查，也可以继续使用 `tests/smoke_test.sh`。

## 本地一行命令跑完
```bash
micromamba run -n calc python run_pipeline.py \
  --upper /abs/path/to/upper.stl \
  --lower /abs/path/to/lower.stl \
  --output /abs/path/to/result.json
```
等价地，也可以直接执行 `./run.sh --upper ... --lower ...`，脚本会自动查找 micromamba。

默认只写出 landmark 字典（结构类似 `dots.json`）。如需保留中间产物（VTP/JSON 和工作目录），加上 `--keep`：
```bash
micromamba run -n calc python run_pipeline.py \
  --upper /data/319_U.stl \
  --lower /data/319_L.stl \
  --output /tmp/319_result.json \
  --keep --pretty
```
此时输出 JSON 会包含：
```json
{
  "result": { "...牙位...": [x, y, z], "...": [...] },
  "artifacts": {
    "workdir": "/absolute/path/to/runs/<uuid>",
    "vtp_a": ".../A.vtp",
    "vtp_b": ".../B.vtp",
    "json_a": ".../A.json",
    "json_b": ".../B.json"
  }
}
```

## 其它说明
- 脚本无额外日志，失败时会在 stderr 打印错误并返回非 0。
- 中间文件默认保存在 `API/runs/<uuid>/`，`--keep` 关闭自动清理，方便自行检查。
- 如果需要计算指标，可直接调用 `utils.runner.run_pipeline(..., run_metrics=True)` 或 `wrappers/calc_metrics_wrapper.py`。
- 推荐把上述命令写入 `start.sh`/`make` 之类的包装脚本，方便他人调用。
