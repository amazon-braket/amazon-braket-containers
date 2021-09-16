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
import os
import time
from contextlib import redirect_stdout

import boto3

from braket.aws import AwsQuantumJob, AwsSession


def job_test(account, role, s3_bucket, image_path, job_type, **kwargs):
    job_output = io.StringIO()
    with redirect_stdout(job_output):
        try:
            create_job(account, role, s3_bucket, image_path, job_type, **kwargs)
        except Exception as e:
            print(e)
    output = job_output.getvalue()
    print(output)
    assert output.find("Braket Container Run Success") > 0, "Container did not run successfully"


def create_job(account, role, s3_bucket, image_path, job_type, **kwargs):
    job_name = f"ContainerTest-{job_type}-{int(time.time())}"
    AwsQuantumJob.create(
        job_name=job_name,
        device_arn="arn:aws:braket:::device/quantum-simulator/amazon/sv1",
        role_arn=f"arn:aws:iam::{account}:role/{role}",
        code_location=f"s3://{s3_bucket}/{job_name}/source",
        image_uri=image_path,
        wait_until_complete=True,
        **kwargs
    )
