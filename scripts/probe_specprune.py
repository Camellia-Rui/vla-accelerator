"""Compatibility entry for the experimental SpecPrune evaluator."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from methods.specprune.evaluation import main
if __name__=='__main__': raise SystemExit(main())
