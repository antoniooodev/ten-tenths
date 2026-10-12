"""Command-line scripts and the demo: they start, and a short evaluate -> analyze run works end to end."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import REPO

SCRIPTS = sorted((REPO / "scripts").glob("*.py"))
ENV = {**os.environ, "PYTHONPATH": str(REPO / "src"), "MPLBACKEND": "Agg"}


def _run(*args: str | Path) -> subprocess.CompletedProcess:
    result = subprocess.run([sys.executable, *map(str, args)], cwd=REPO, env=ENV, capture_output=True, text=True,
                            timeout=900)
    assert result.returncode == 0, result.stderr[-3000:]
    return result


@pytest.mark.parametrize("script", [s for s in SCRIPTS if s.name != "make_figures.py"], ids=lambda p: p.stem)
def test_help(script: Path) -> None:
    assert "usage" in _run(script, "--help").stdout


def test_evaluate_then_analyze(tmp_path: Path) -> None:
    protocol = json.loads((REPO / "data/protocols/catalunya_R1.json").read_text())
    protocol["episodes"] = [dict(e, max_steps=200) for e in protocol["episodes"][:2]]
    protocol["racing_line_file"] = str(REPO / protocol["racing_line_file"])
    path = tmp_path / "catalunya_R1.json"
    path.write_text(json.dumps(protocol))
    for who in (("--model", REPO / "models/all4"), ("--teacher",)):
        for condition in ("U", "F"):
            name = "all4" if who[0] == "--model" else "teacher"
            _run("scripts/evaluate.py", "--protocol", path, *who, "--condition", condition,
                 "--output", tmp_path / f"{name}_{condition}.jsonl")
    _run("scripts/analyze.py", "--student", tmp_path / "all4", "--teacher", tmp_path / "teacher",
         "--output", tmp_path / "summary")
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["student"]["U"]["n"] == 2 and summary["teacher"]["F"]["n"] == 2
    assert summary["filter_student"]["paired"] == 2


def test_demo_quick(tmp_path: Path) -> None:
    out = _run("examples/demo.py", "--max-steps", "200", "--print-every", "0", "--json", tmp_path / "demo.json")
    records = json.loads((tmp_path / "demo.json").read_text())
    assert {r["scenario"] for r in records} == {"circuits", "zero-shot", "filter", "friction", "racing-line"}
    assert {r["circuit"] for r in records} == {"monza", "spa", "catalunya", "budapest"}
    assert "scenario" in out.stdout


def test_demo_racing_line_kink() -> None:
    """LOCO Spa from start h8 at mu 1.2 leaves the track at the R0 kink (recorded: 3035 steps; 2933 with the sides of
    the track boundaries swapped, before the fix of docs/REDESIGN.md, section 17)."""
    out = _run("examples/demo.py", "--circuit", "spa", "--policy", "loco_spa", "--line", "R0", "--start", "h8",
               "--mu", "1.2", "--print-every", "0")
    assert "track_excursion in zone T1 after 3035 steps" in out.stdout
