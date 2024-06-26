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

import io
import sys
import time
from contextlib import redirect_stdout

from braket.aws import AwsSession
from braket.jobs import hybrid_job
from braket.devices import Devices

from ...resources.qaoa_entry_point import entry_point


def job_test(account, role, s3_bucket, image_path, use_local_jobs, use_local_sim, job_type, interface):
    job_output = io.StringIO()
    with redirect_stdout(job_output):
        try:
            job_args = {
                "p": 2,
                "seed": 1967,
                "max_parallel": 10,
                "num_iterations": 5,
                "stepsize": 0.1,
                "shots": 100,
                "pl_interface": interface,
                "start_time": time.time(),
            }
            create_job(account, role, s3_bucket, image_path, use_local_jobs, use_local_sim, job_type, job_args)
        except Exception as e:
            print(e)
    output = job_output.getvalue()
    print(output)
    assert output.find("Braket Container Run Success") > 0, "Container did not run successfully"


def create_job(account, role, s3_bucket, image_path, use_local_jobs, use_local_sim, job_type, job_args):
    aws_session = AwsSession(default_bucket=s3_bucket)
    job_name = f"ContainerTest-{job_type}-{int(time.time())}"

    @hybrid_job(
        aws_session=aws_session,
        job_name=job_name,
        device="local:none/none" if use_local_sim else Devices.Amazon.SV1,
        role_arn=f"arn:aws:iam::{account}:role/{role}",
        image_uri=image_path,
        wait_until_complete=True,
        local=use_local_jobs,
        include_modules="test.resources",
    )
    def decorator_job(*args, **kwargs):
        return entry_point(*args, **kwargs)

    decorator_job(**job_args)
