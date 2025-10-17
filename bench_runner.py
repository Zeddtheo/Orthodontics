import time
from pathlib import Path
from API.utils.runner import run_pipeline
start = time.perf_counter()
res = run_pipeline(Path("API/tests/3_U.stl"), Path("API/tests/3_L.stl"))
elapsed = time.perf_counter() - start
print(f"duration={elapsed:.2f}s")
print(res)
