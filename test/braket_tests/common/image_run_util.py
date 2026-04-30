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

import subprocess
from dataclasses import dataclass


@dataclass
class RunResult:
    returncode: int
    stdout: str
    stderr: str

    @property
    def combined(self) -> str:
        return self.stdout + self.stderr


def run_in_image(image_path: str, command: list, timeout: int = 60) -> RunResult:
    """Run a command inside the given image via `docker run --rm` and return the result.

    The image is expected to already be present locally (pulled from ECR by the
    caller, or built locally). `command` is passed as the container's argv; the
    caller supplies the interpreter (e.g. ["bash", "-lc", "mpirun --version"]).
    """
    proc = subprocess.run(
        ["docker", "run", "--rm", "--entrypoint", "", image_path, *command],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return RunResult(
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )
