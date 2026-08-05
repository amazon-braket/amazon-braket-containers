"""The cudaq image is built independently of the base image, so it duplicates
some of base's setup. These tests keep the copies from drifting: regions marked
``# shared-block:begin <name>`` / ``# shared-block:end <name>`` must be
byte-identical across the Dockerfiles below.
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

DOCKERFILES = (
    "base/jobs/docker/1.0/py3/Dockerfile.cpu",
    "cudaq/jobs/docker/0.14/py3/Dockerfile.cpu",
)

# Fails if a block is renamed or its markers are deleted from every Dockerfile.
EXPECTED_BLOCKS = {"runtime-env", "system-deps"}

BEGIN = re.compile(r"^#\s*shared-block:begin\s+(\S+)\s*$")
END = re.compile(r"^#\s*shared-block:end\s+(\S+)\s*$")


def _parse_blocks(path):
    """Map block name -> list of lines between its begin/end markers."""
    blocks = {}
    open_name = None
    collected = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        begin = BEGIN.match(line)
        end = END.match(line)
        if begin:
            assert open_name is None, (
                f"{path}:{lineno}: shared-block:begin {begin.group(1)} while "
                f"{open_name} is still open"
            )
            open_name = begin.group(1)
            collected = []
        elif end:
            assert open_name == end.group(1), (
                f"{path}:{lineno}: shared-block:end {end.group(1)} does not "
                f"match open block {open_name}"
            )
            assert open_name not in blocks, (
                f"{path}:{lineno}: duplicate shared block {open_name}"
            )
            assert collected, f"{path}:{lineno}: shared block {open_name} is empty"
            blocks[open_name] = collected
            open_name = None
        elif open_name is not None:
            collected.append(line)
    assert open_name is None, f"{path}: shared block {open_name} is never closed"
    return blocks


@pytest.fixture(scope="module")
def parsed_blocks():
    return {name: _parse_blocks(REPO_ROOT / name) for name in DOCKERFILES}


@pytest.mark.parametrize("dockerfile", DOCKERFILES)
def test_dockerfile_declares_expected_shared_blocks(parsed_blocks, dockerfile):
    assert set(parsed_blocks[dockerfile]) == EXPECTED_BLOCKS


@pytest.mark.parametrize("block_name", sorted(EXPECTED_BLOCKS))
def test_shared_blocks_are_identical_across_dockerfiles(parsed_blocks, block_name):
    reference, *others = DOCKERFILES
    for other in others:
        assert parsed_blocks[other][block_name] == parsed_blocks[reference][block_name], (
            f"shared block {block_name!r} differs between {reference} and {other}. "
            f"Apply the change to both copies."
        )
