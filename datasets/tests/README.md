# 测试数据目录说明

本目录包含 316–320 五个测试病例。每个子文件夹内主要文件如下：
原始数据
- `<病例号>_U.stl` / `<病例号>_L.stl`：原始上颌 / 下颌 STL 网格。
- `<病例号>_U.json` / `<病例号>_L.json`：医生人工标注的上颌 / 下颌标点。
- 预测数据
- `api_results/`：API 批处理输出
  - `<病例号>_U_seg.vtp` / `<病例号>_L_seg.vtp`：ios-model 分割后的 VTP 网格。
  - `<病例号>_U_pred.json` / `<病例号>_L_pred.json`：PointNet-Reg 预测的标点。
  - `<病例号>_report.md`：整合 calc_metrics 指标与误差统计的报告。
