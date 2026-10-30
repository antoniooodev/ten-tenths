"""The figures of the README, in a light and a dark theme, from make_thesis_figures.py.

    python scripts/figures/make_readme_figures.py
    python scripts/figures/make_readme_figures.py --only gg_diagram

Renders the figures of ``README_FIGURES`` without their titles (the README captions them) once per theme, each theme
in its own process (thesis_plots reads TEN_TENTHS_THEME at import), and copies the PNG files to
docs/figures/light/<name>.png and docs/figures/dark/<name>.png. The README shows the one that matches the reader's
GitHub theme with a <picture> element.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# figure name in make_thesis_figures.py -> file name in docs/figures/<theme>/
README_FIGURES = {
    "vehicle_model": "vehicle_model",
    "tyre_model": "tyre_model",
    "balance_handling": "balance_handling",
    "track_layout": "track_layout",
    "optimal_line": "optimal_line",
    "training_nominal": "training_nominal",
    "lap_profiles": "lap_profiles",
    "corner_zooms": "corner_zooms",
    "gg_diagram": "gg_diagram",
    "tyre_usage": "tyre_usage",
    "protocol_gaps": "protocol_gaps",
    "balance_heatmaps": "balance_heatmaps",
    "balance_corner": "balance_corner",
    "mean_vs_stochastic": "mean_vs_stochastic",
    "cost_and_regret": "cost_and_regret",
}
DPI = 160


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", choices=sorted(README_FIGURES))
    parser.add_argument("--themes", nargs="+", default=["light", "dark"], choices=["light", "dark"])
    parser.add_argument("--work", type=Path, default=Path("outputs/figures/readme"))
    parser.add_argument("--docs", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()
    names = args.only or list(README_FIGURES)
    script = Path(__file__).with_name("make_thesis_figures.py")
    for theme in args.themes:
        work = args.work / theme
        env = {**os.environ, "TEN_TENTHS_THEME": theme}
        subprocess.run([sys.executable, str(script), "--no-titles", "--dpi", str(DPI), "--output", str(work),
                        "--only", *names], check=True, env=env)
        target = args.docs / theme
        target.mkdir(parents=True, exist_ok=True)
        for name in names:
            (source,) = work.glob(f"*_{name}.png")
            shutil.copyfile(source, target / f"{README_FIGURES[name]}.png")
        print(f"{theme}: {len(names)} figures -> {target}")


if __name__ == "__main__":
    main()
