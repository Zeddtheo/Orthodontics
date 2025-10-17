# 10-case Evaluation

- PointNet-Reg output directory: `C:\MISC\Deepcare\Orthodontics\outputs\seg_point_eval\pointnet`
- Segmentation staging raw: `C:\MISC\Deepcare\Orthodontics\outputs\seg_point_eval\raw`

| Case | Matched / GT | MAE (mm) | P95 (mm) | Hit@0.5 | Hit@1.0 |
| --- | --- | --- | --- | --- | --- |
| 29 | 242 / 242 | 0.294 | 0.687 | 85.124% | 98.347% |
| 44 | 244 / 244 | 0.325 | 0.742 | 83.607% | 99.180% |
| 54 | 244 / 244 | 0.272 | 0.790 | 87.295% | 97.951% |
| 58 | 244 / 244 | 0.538 | 1.514 | 72.541% | 92.623% |
| 97 | 244 / 244 | 0.474 | 1.002 | 78.279% | 94.672% |

## Aggregated

- Mean MAE: 0.380 mm
- Median MAE: 0.325 mm
- Std MAE: 0.106 mm
- Mean P95: 0.947 mm
- Hit@0.5 avg: 81.369%
- Hit@1.0 avg: 96.555%