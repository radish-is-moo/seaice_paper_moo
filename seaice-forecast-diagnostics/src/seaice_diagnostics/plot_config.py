from .config import REPO, RESULTS, FIGURES
import os
from pathlib import Path

INPUT = Path(os.environ.get("SEAICE_PLOT_INPUT", REPO / "data/figure_inputs")).resolve()
OCEAN = INPUT / "ocean"
STATES = INPUT / "states"
INITIAL = INPUT / "initial_conditions"
