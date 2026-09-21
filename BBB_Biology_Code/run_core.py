import subprocess
import sys
from pathlib import Path

S = Path(__file__).resolve().parent / "scripts"
STEPS = [
    ["stage0_audit_v15.py"], ["stage5_consensus.py"], ["stage05_conflict_audit.py"], ["stage_e2_multiverse.py"],
    ["stage7_scaffold_split.py"], ["stage7_freeze_eval.py"], ["stage6_hem.py"], ["stage7_main.py"], ["stage7_gine.py"],
    ["stage7_bootstrap.py", "xgboost"], ["stage7_bootstrap.py", "gine"], ["stage_e3_uncertainty.py"],
    ["stage_external_science_v13.py"], ["stage_external_science_v15.py"]
]
for step in STEPS:
    subprocess.run([sys.executable, str(S / step[0]), *step[1:]], check=True)
