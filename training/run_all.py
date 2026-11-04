"""Execute the training cells in order in one namespace (the same code runs as notebook cells on Colab).

    python training/run_all.py              # trains every run whose done.json is missing
    ISICLEAK_SMOKE=1 python training/run_all.py   # pipeline test on a tiny subset
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
namespace = {"__name__": "__main__"}
for cell in sorted(HERE.glob("0*.py")):
    print(f"\n#### {cell.name}")
    exec(compile(cell.read_text(encoding="utf-8"), str(cell), "exec"), namespace)
