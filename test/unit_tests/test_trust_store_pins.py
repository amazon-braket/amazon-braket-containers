"""Verify the TensorFlow image trust-store fix."""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIR = REPO_ROOT / "tensorflow/jobs/docker/2.19/py3"
CERTIFI_PIN = "certifi==2026.10.19"
PIP_PIN = "pip==26.2.2"
PIP_WHEEL = "pip-26.2.2-py3-none-any.whl"
PIP_WHEEL_SHA256 = "fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70"


def _requirements():
    lines = (IMAGE_DIR / "requirements.txt").read_text().splitlines()
    return [line.split("#", 1)[0].strip() for line in lines]


def _dockerfile():
    return (IMAGE_DIR / "Dockerfile.gpu").read_text()


def test_requirements_pin_trust_stores():
    requirements = _requirements()
    assert CERTIFI_PIN in requirements
    assert PIP_PIN in requirements


def test_dockerfile_refreshes_os_ca_certificates():
    commands = re.sub(r"\\\n", " ", _dockerfile())
    assert re.search(
        r"apt-get update.*apt-get install[^\n]*\bca-certificates\b", commands
    )
    assert "update-ca-certificates" in commands


def test_dockerfile_bootstraps_pip_from_verified_wheel():
    dockerfile = _dockerfile()
    assert PIP_WHEEL in dockerfile
    assert PIP_WHEEL_SHA256 in dockerfile
    assert "sha256sum --check" in dockerfile
    assert "--no-index --no-deps" in dockerfile
