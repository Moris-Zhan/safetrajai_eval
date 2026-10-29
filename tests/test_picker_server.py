"""Token-protected web service that serves the picker pages on demand."""

import threading
import urllib.error
import urllib.request

import pytest

from safetrajai.picker.build import PickerInputError
from safetrajai.picker.server import make_server

TOKEN = "secret-token"


def start(build, cameras):
    server = make_server("127.0.0.1", 0, TOKEN, build, cameras)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def stop(server) -> None:
    server.shutdown()
    server.server_close()


@pytest.fixture
def base_url():
    calls = []

    def build(camera: str, mode: str) -> str:
        calls.append((camera, mode))
        return f"<html>page {camera} {mode}</html>"

    server, url = start(build, lambda: ["C000025", "C000054"])
    yield url, calls
    stop(server)


def get(url: str, headers: dict | None = None):
    request = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, response.read().decode("utf-8"), response.headers
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8"), error.headers


def test_requests_without_the_right_token_are_refused_and_build_nothing(base_url) -> None:
    url, calls = base_url

    assert get(f"{url}/")[0] == 401
    assert get(f"{url}/?token=wrong")[0] == 401
    assert get(f"{url}/page/C000025/homography")[0] == 401
    assert get(f"{url}/page/C000025/homography?token=wrong")[0] == 401
    assert calls == []


def test_index_lists_every_camera_and_mode_with_links(base_url) -> None:
    url, _ = base_url

    status, body, headers = get(f"{url}/?token={TOKEN}")

    assert status == 200 and "C000025" in body and "C000054" in body
    for mode in ("homography", "checkpoints", "counting-lines", "camera-position"):
        assert f"/page/C000025/{mode}" in body
    assert "no-store" in headers["Cache-Control"]


def test_a_page_is_built_on_demand(base_url) -> None:
    url, calls = base_url

    status, body, _ = get(f"{url}/page/C000025/counting-lines?token={TOKEN}")

    assert (status, body) == (200, "<html>page C000025 counting-lines</html>")
    assert calls == [("C000025", "counting-lines")]


def test_the_cookie_set_by_a_valid_token_opens_later_requests(base_url) -> None:
    url, _ = base_url

    _, _, headers = get(f"{url}/?token={TOKEN}")
    cookie = headers["Set-Cookie"].split(";")[0]

    assert "HttpOnly" in headers["Set-Cookie"]
    assert get(f"{url}/page/C000025/homography", {"Cookie": cookie})[0] == 200


@pytest.mark.parametrize("path", [
    "/page/C000025/roi", "/page/C000025", "/page/../etc/passwd/homography", "/page/C0000%2f25/homography",
    "/page/C000077/homography", "/nothing"])
def test_unknown_routes_cameras_and_modes_are_404_without_building(base_url, path: str) -> None:
    url, calls = base_url

    assert get(f"{url}{path}?token={TOKEN}")[0] == 404
    assert calls == []


def test_build_errors_of_a_listed_camera_are_400_with_the_escaped_reason() -> None:
    def build(camera: str, mode: str) -> str:
        raise PickerInputError("找不到衛星圖 <x>")

    server, url = start(build, lambda: ["C000025"])
    try:
        status, body, _ = get(f"{url}/page/C000025/homography?token={TOKEN}")
    finally:
        stop(server)

    assert status == 400 and "找不到衛星圖 &lt;x&gt;" in body


def test_an_empty_token_is_rejected() -> None:
    with pytest.raises(ValueError, match="token"):
        make_server("127.0.0.1", 0, "", lambda c, m: "", lambda: [])


def test_token_is_created_once_kept_private_and_reused(tmp_path) -> None:
    from safetrajai.picker.server import load_or_create_token, redact_token

    path = tmp_path / "runs" / "picker" / "server_token"

    first = load_or_create_token(path)

    assert len(first) >= 20 and load_or_create_token(path) == first
    assert path.stat().st_mode & 0o777 == 0o600
    assert redact_token(f"GET /?token={first}&x=1 HTTP/1.1") == "GET /?token=***&x=1 HTTP/1.1"
