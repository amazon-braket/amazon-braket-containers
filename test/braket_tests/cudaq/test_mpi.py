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

from ..common.image_run_util import run_in_image


# CUDA-Q's activate_custom_mpi.sh compiles this shared object against the
# base image's MPI headers at image-build time. Its absence means the plugin
# failed to build (silently, since the RUN step would have errored loudly) or
# was installed to an unexpected path.
CUDAQ_MPI_PLUGIN_PATH = (
    "/usr/local/lib/python3.12/site-packages/distributed_interfaces/"
    "libcudaq_distributed_interface_mpi.so"
)


def test_mpi_init_no_crash_default_env(image_list):
    """Regression test: when OMPI_MCA_opal_cuda_support=true was set in the
    CUDA-Q Dockerfile but the base OpenMPI was not built with --with-cuda,
    MPI_Init would emit `opal_cuda_support` errors and segfault during
    library initialization on non-GPU hosts. This runs a trivial MPI library
    probe in the image's default env and asserts clean success.
    """
    assert len(image_list) > 0, "Unable to find images for testing"
    for image_path in image_list:
        result = run_in_image(
            image_path,
            [
                "python",
                "-c",
                "from mpi4py import MPI; print(MPI.Get_library_version().strip())",
            ],
        )
        assert result.returncode == 0, (
            f"mpi4py MPI_Init crashed in {image_path} (exit {result.returncode}).\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        combined_lower = result.combined.lower()
        for bad in ("segmentation fault", "opal_cuda_support", "mpi_cuda_support"):
            assert bad not in combined_lower, (
                f"Unexpected MPI error string {bad!r} in {image_path} output:\n"
                f"{result.combined}"
            )


def test_cudaq_mpi_plugin_present(image_list):
    """CUDA-Q's custom MPI plugin is compiled by activate_custom_mpi.sh during
    image build. If the plugin is missing, CUDA-Q's distributed MPI interface
    will fall back to a stub and silently run as a single rank.
    """
    assert len(image_list) > 0, "Unable to find images for testing"
    for image_path in image_list:
        result = run_in_image(image_path, ["test", "-f", CUDAQ_MPI_PLUGIN_PATH])
        assert result.returncode == 0, (
            f"CUDA-Q MPI plugin missing at {CUDAQ_MPI_PLUGIN_PATH} in {image_path}. "
            f"activate_custom_mpi.sh likely failed silently during image build."
        )


def test_cudaq_mpi_initialize_finalize(image_list):
    """End-to-end check that CUDA-Q's MPI subsystem initializes against the
    base image's OpenMPI. Covers linkage of libcudaq_distributed_interface_mpi.so
    in addition to basic MPI runtime sanity.
    """
    assert len(image_list) > 0, "Unable to find images for testing"
    script = (
        "import cudaq; "
        "cudaq.mpi.initialize(); "
        "assert cudaq.mpi.rank() == 0; "
        "assert cudaq.mpi.num_ranks() == 1; "
        "cudaq.mpi.finalize()"
    )
    for image_path in image_list:
        result = run_in_image(image_path, ["python", "-c", script])
        assert result.returncode == 0, (
            f"cudaq.mpi initialize/finalize failed in {image_path} "
            f"(exit {result.returncode}).\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
