# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

"""Fast smoke tests for OpenMPI configuration baked into the base image.

These tests run `docker run` one-liners (no AWS, no SageMaker), so they complete
in a few seconds per image. They guard against regressions in the OpenMPI build
and MCA parameter configuration in base/jobs/docker/1.0/py3/Dockerfile.cpu.
"""

import re

from ..common.image_run_util import run_in_image


# Strictly pin to OpenMPI 4.1.x.
#
# This is deliberately stricter than "matches whatever is in the Dockerfile":
# - 5.x renamed ORTE -> PRRTE and changed several runtime defaults; an accidental
#   bump would break consumers that rely on 4.1 behavior (e.g. CUDA-Q's custom MPI
#   plugin was validated against 4.1.8).
# - Any 4.0.x version is EOL/retired and would be a downgrade bug.
# Updating this assertion should be a deliberate, reviewed change.
EXPECTED_OPENMPI_MAJOR_MINOR = "4.1"


def test_openmpi_version_pinned(image_list):
    assert len(image_list) > 0, "Unable to find images for testing"
    for image_path in image_list:
        result = run_in_image(image_path, ["mpirun", "--version"])
        assert result.returncode == 0, (
            f"mpirun --version exited {result.returncode}: {result.combined}"
        )
        # Output looks like: "mpirun.real (OpenRTE) 4.1.8"
        match = re.search(r"\(Open(?:RTE|MPI|-MPI)\)\s+(\d+)\.(\d+)\.(\d+)", result.combined)
        assert match, f"Could not parse OpenMPI version from: {result.combined!r}"
        major_minor = f"{match.group(1)}.{match.group(2)}"
        assert major_minor == EXPECTED_OPENMPI_MAJOR_MINOR, (
            f"OpenMPI {major_minor}.{match.group(3)} detected in {image_path}; "
            f"expected {EXPECTED_OPENMPI_MAJOR_MINOR}.x. "
            f"If this is an intentional version change, update "
            f"EXPECTED_OPENMPI_MAJOR_MINOR in this test."
        )


def test_mpi_no_keyval_parser_errors(image_list):
    """Regression test: invalid `echo -e` usage in the base Dockerfile once wrote
    a literal `-e ` prefix into openmpi-mca-params.conf, causing every MPI
    invocation to emit `keyval parser: error ...` warnings. This asserts no such
    warnings surface when running a trivial mpirun invocation in the default env.
    """
    assert len(image_list) > 0, "Unable to find images for testing"
    for image_path in image_list:
        result = run_in_image(image_path, ["mpirun", "--version"])
        assert "keyval parser: error" not in result.combined, (
            f"MPI keyval parser errors detected in {image_path}; "
            f"openmpi-mca-params.conf is likely malformed. Output:\n{result.combined}"
        )


def test_mpi_mca_params_applied(image_list):
    """Both default MCA policies declared in the base Dockerfile must be loaded
    from the system config file at runtime. This guards the *intent* (both
    policies take effect) independent of how they're written in the Dockerfile,
    so it will catch e.g. only one of the two being persisted.
    """
    assert len(image_list) > 0, "Unable to find images for testing"
    for image_path in image_list:
        result = run_in_image(
            image_path,
            ["ompi_info", "--param", "all", "all", "--level", "9"],
            timeout=90,
        )
        assert result.returncode == 0, (
            f"ompi_info exited {result.returncode}: {result.combined}"
        )
        # Expect each parameter reported with current value and source file.
        for param, value in (
            ("hwloc_base_binding_policy", "none"),
            ("rmaps_base_mapping_policy", "slot"),
        ):
            pattern = rf'parameter "{param}" \(current value: "{value}".*data source: file'
            assert re.search(pattern, result.stdout), (
                f'Expected MCA param {param}={value} loaded from config file '
                f"in {image_path}, but did not find it in ompi_info output."
            )
