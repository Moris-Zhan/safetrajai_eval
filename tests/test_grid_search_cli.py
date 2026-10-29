"""Command-line surface of the three grid search scripts (snapshot of --help, --dry-run and error exits)."""

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = Path(__file__).parent / "fixtures" / "grid_search_cli"
SCRIPTS = ("grid_search_bytetrack", "grid_search_botsort", "grid_search_botsort_reid")
CASES = {
    "help": ["--help"],
    "dry_run_subset": ["--target-class", "car", "--params-subset", "fuse_score", "--dry-run"],
    "unknown_subset": ["--target-class", "car", "--params-subset", "no_such_param", "--dry-run"],
}


def run_case(script: str, args: list[str]) -> dict:
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "grid_search" / f"{script}.py"), *args],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=300,
    )
    lines = [line for line in result.stdout.replace(str(REPO_ROOT), "<root>").splitlines()
             if not line.startswith("Error importing BURST")]
    return {"returncode": result.returncode, "stdout": "\n".join(lines)}


@pytest.fixture(scope="module")
def outputs() -> dict[tuple[str, str], dict]:
    jobs = [(script, case) for script in SCRIPTS for case in CASES]
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        results = list(pool.map(lambda job: run_case(job[0], CASES[job[1]]), jobs))
    return dict(zip(jobs, results))


@pytest.mark.parametrize("script", SCRIPTS)
@pytest.mark.parametrize("case", CASES)
def test_cli_output_matches_snapshot(outputs: dict, script: str, case: str) -> None:
    expected = json.loads((SNAPSHOT_DIR / f"{script}.json").read_text(encoding="utf-8"))[case]
    assert outputs[(script, case)] == expected
