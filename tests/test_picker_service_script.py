"""scripts/picker/start_picker_service.sh: background start that does nothing when the service already answers."""

import os
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from safetrajai.paths import PROJECT_ROOT

SCRIPT = PROJECT_ROOT / "scripts" / "picker" / "start_picker_service.sh"
needs_tools = pytest.mark.skipif(shutil.which("conda") is None or shutil.which("curl") is None,
                                 reason="conda and curl are needed to start the service")


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def status(port: int) -> int | None:
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2)
    except urllib.error.HTTPError as error:
        return error.code
    except OSError:
        return None
    return 200


def test_script_is_valid_bash() -> None:
    assert subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True).returncode == 0


@needs_tools
def test_service_starts_once_and_a_second_call_leaves_it_alone(tmp_path: Path) -> None:
    port = free_port()
    env = {**os.environ, "PICKER_PORT": str(port), "PICKER_LOG": str(tmp_path / "server.log")}
    try:
        first = subprocess.run(["bash", str(SCRIPT)], capture_output=True, text=True, env=env, timeout=60)
        deadline = time.time() + 90
        while status(port) is None and time.time() < deadline:
            time.sleep(1)
        second = subprocess.run(["bash", str(SCRIPT)], capture_output=True, text=True, env=env, timeout=60)

        assert first.returncode == 0 and "started" in first.stdout
        assert status(port) == 401                      # answers, and refuses requests without the token
        assert second.returncode == 0 and "already running" in second.stdout
    finally:
        subprocess.run(["pkill", "-f", f"serve_picker.py --port {port}"], check=False)
