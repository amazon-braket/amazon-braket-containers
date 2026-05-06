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

"""Fast smoke tests for MPI behavior in the CUDA-Q image.

These tests run `docker run` one-liners (no AWS, no SageMaker, no GPU), so they
complete in a few seconds per image. They guard against regressions in how the
CUDA-Q image composes on top of the base image's OpenMPI build.
"""

from ..common.mpi_checks import (
    assert_cudaq_mpi_initialize_finalize,
    assert_cudaq_mpi_plugin_present,
    assert_mpi4py_init_no_crash,
    assert_mpirun_multirank_bcast,
)
from ..common.image_run_util import run_in_image


def test_mpi_init_no_crash_default_env(image_list):
    """Regression test: when OMPI_MCA_opal_cuda_support=true was set in the
    CUDA-Q Dockerfile but the base OpenMPI was not built with --with-cuda,
    MPI_Init would emit `opal_cuda_support` errors and segfault during
    library initialization on non-GPU hosts. Verifies CUDA-Q's image build
    didn't reintroduce that (or a similar) misconfiguration on top of base.
    """
    assert_mpi4py_init_no_crash(image_list)


def test_cudaq_mpi_plugin_present(image_list):
    """CUDA-Q's custom MPI plugin is compiled by activate_custom_mpi.sh during
    image build against the base image's from-source OpenMPI. If the plugin is
    missing, CUDA-Q's distributed MPI interface will fall back to a stub and
    silently run as a single rank.
    """
    assert_cudaq_mpi_plugin_present(image_list)


def test_cudaq_mpi_initialize_finalize(image_list):
    """End-to-end check that CUDA-Q's MPI subsystem initializes against the
    base image's OpenMPI. Covers linkage of libcudaq_distributed_interface_mpi.so
    in addition to basic MPI runtime sanity.
    """
    assert_cudaq_mpi_initialize_finalize(image_list)


def test_cuda_aware_mpi_built(image_list):
    """OpenMPI must be built with --with-cuda for cuStateVec mgpu support.
    Without CUDA-aware MPI, cuStateVec segfaults when passing GPU device
    pointers to MPI_Isend/MPI_Irecv during distributed state vector swaps.
    """
    assert len(image_list) > 0, "Unable to find images for testing"
    for image_path in image_list:
        result = run_in_image(
            image_path,
            [
                "bash", "-c",
                "ompi_info --parsable --all | grep mpi_built_with_cuda_support:value:true",
            ],
        )
        assert result.returncode == 0, (
            f"OpenMPI not built with CUDA support in {image_path}.\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def test_mpirun_multirank_bcast(image_list):
    """Multi-rank MPI communication must work in the CUDA-Q container."""
    assert_mpirun_multirank_bcast(image_list)
