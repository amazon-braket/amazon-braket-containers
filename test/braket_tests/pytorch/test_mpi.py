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

"""Fast smoke tests for MPI behavior in the PyTorch GPU image.

Unlike the base image (where this repo builds OpenMPI from source), the PyTorch
image inherits its MPI installation from the AWS Deep Learning Container base
image. These tests therefore do not pin a specific MPI version; they verify
that MPI is present and that mpirun works in the image's default environment.

Actual GPU-to-GPU MPI messaging would require --gpus all + an NVIDIA driver
on the test host and is out of scope for these smoke tests. Note that the
DLC's OpenMPI is intentionally NOT built with --with-cuda; the DLC uses
libfabric/EFA + NCCL for high-bandwidth GPU collectives, not CUDA-aware MPI.
"""

from ..common.image_run_util import run_in_image
from ..common.mpi_checks import (
    assert_cudaq_mpi_initialize_finalize,
    assert_cudaq_mpi_plugin_present,
    assert_mpi_runtime_launches_cleanly,
)


def test_mpirun_available(image_list):
    assert len(image_list) > 0, "Unable to find images for testing"
    for image_path in image_list:
        result = run_in_image(image_path, ["bash", "-lc", "mpirun --version"])
        assert result.returncode == 0, (
            f"mpirun --version exited {result.returncode} in {image_path}: "
            f"{result.combined}"
        )


def test_mpi_runtime_launches_cleanly(image_list):
    """Regression test mirroring the base/cudaq image checks: `mpirun` must be
    able to launch a trivial process in the image's default environment without
    emitting errors or crashing. Catches any Dockerfile change that sets an MCA
    flag inconsistent with how the underlying OpenMPI was built.

    Uses /bin/true so we don't depend on mpi4py being installed in the DLC.
    """
    assert_mpi_runtime_launches_cleanly(image_list)


def test_cudaq_mpi_plugin_present(image_list):
    """The pytorch image installs cudaq from pip but must also run
    activate_custom_mpi.sh during image build to compile the MPI plugin
    against the DLC's OpenMPI under /opt/amazon/openmpi. Without this,
    cudaq.mpi APIs raise ``RuntimeError: No MPI support can be found``.
    """
    assert_cudaq_mpi_plugin_present(image_list)


def test_cudaq_mpi_initialize_finalize(image_list):
    """End-to-end check that CUDA-Q's MPI subsystem initializes and finalizes
    cleanly in the pytorch image. Specifically guards against regressions where
    the cudaq pip install is present but its custom MPI plugin isn't wired up
    against the DLC's OpenMPI.
    """
    assert_cudaq_mpi_initialize_finalize(image_list)
