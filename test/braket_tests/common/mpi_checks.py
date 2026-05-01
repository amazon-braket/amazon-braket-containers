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

"""Shared MPI smoke-test helpers used by multiple image profiles.

Each helper runs a trivial check inside every image in ``image_list`` and
raises AssertionError on failure, so callers in profile-specific test modules
can stay a one-liner while keeping their own test names and docstrings to
document intent.
"""

from .image_run_util import run_in_image


def assert_mpi_runtime_launches_cleanly(image_list):
    """Assert ``mpirun -n 1 /bin/true`` succeeds cleanly in each image.

    Catches Dockerfile changes that set MCA flags inconsistent with how the
    underlying OpenMPI was built (e.g. enabling a transport or CUDA support
    that wasn't compiled in), which typically surface as a segfault during
    library init or a ``was not compiled with`` error string.

    Uses /bin/true so this helper has no Python-level MPI dependency and can
    run against images that don't ship mpi4py.
    """
    assert len(image_list) > 0, "Unable to find images for testing"
    for image_path in image_list:
        result = run_in_image(
            image_path,
            ["bash", "-lc", "mpirun -n 1 --allow-run-as-root /bin/true"],
        )
        assert result.returncode == 0, (
            f"mpirun failed in {image_path} (exit {result.returncode}).\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        combined_lower = result.combined.lower()
        for bad in ("segmentation fault", "was not compiled with"):
            assert bad not in combined_lower, (
                f"mpirun emitted unexpected error {bad!r} in {image_path}:\n"
                f"{result.combined}"
            )


def assert_mpi4py_init_no_crash(image_list):
    """Assert mpi4py can import and complete ``MPI_Init`` cleanly in each image.

    This is a stronger check than ``assert_mpi_runtime_launches_cleanly`` because
    it actually loads libmpi into a Python process and calls into it. Originally
    added to catch an OMPI_MCA_opal_cuda_support misconfiguration that caused
    MPI_Init to emit ``opal_cuda_support`` errors and segfault on non-GPU hosts.

    Requires mpi4py to be installed in the image.
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


# CUDA-Q's activate_custom_mpi.sh compiles this shared object against whatever
# MPI headers are present in the image at build time (base's from-source OpenMPI
# in the cudaq image, the DLC's OpenMPI under /opt/amazon/openmpi in the pytorch
# image). Both images install cudaq from pip into the same site-packages layout,
# so the plugin ends up at this path either way. Absence of the file means
# activate_custom_mpi.sh did not run or failed silently during image build, in
# which case ``cudaq.mpi`` APIs raise ``RuntimeError: No MPI support can be found``.
CUDAQ_MPI_PLUGIN_PATH = (
    "/usr/local/lib/python3.12/site-packages/distributed_interfaces/"
    "libcudaq_distributed_interface_mpi.so"
)


def assert_cudaq_mpi_plugin_present(image_list):
    """Assert CUDA-Q's custom MPI plugin ``.so`` exists in each image."""
    assert len(image_list) > 0, "Unable to find images for testing"
    for image_path in image_list:
        result = run_in_image(image_path, ["test", "-f", CUDAQ_MPI_PLUGIN_PATH])
        assert result.returncode == 0, (
            f"CUDA-Q MPI plugin missing at {CUDAQ_MPI_PLUGIN_PATH} in {image_path}. "
            f"activate_custom_mpi.sh likely did not run or failed silently during "
            f"image build."
        )


def assert_cudaq_mpi_initialize_finalize(image_list):
    """Assert ``cudaq.mpi.initialize()`` / ``finalize()`` succeed in each image.

    End-to-end check that CUDA-Q's MPI subsystem links against the image's
    OpenMPI correctly. Stronger than the plugin-presence check because it
    actually loads and exercises the plugin.
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
