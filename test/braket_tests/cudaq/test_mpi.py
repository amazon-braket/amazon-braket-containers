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
)


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
