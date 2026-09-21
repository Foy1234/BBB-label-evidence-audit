import subprocess
import sys
from pathlib import Path

S = Path(__file__).resolve().parent / "scripts"
STEPS = [
    ["stage_c3_qsardb_validation.py"], ["stage_e1_provenance.py"], ["stage_c2_cleanlab.py"], ["stage_c4_pretrained.py"],
    ["stage_e4_simulation.py"], ["stage_f2_f7_final_v10.py"], ["stage_calibration_ablation_v13.py"], ["stage_calibration_ablation_v15.py"]
]
for step in STEPS:
    subprocess.run([sys.executable, str(S / step[0]), *step[1:]], check=True)
