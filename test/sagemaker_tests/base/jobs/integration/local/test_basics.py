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

import os
import io
from contextlib import redirect_stdout
import boto3
from botocore.session import get_session

from sagemaker.estimator import Estimator

SCRIPT_NAME = "run_script.py"
SCRIPT_PATH = "./test/resources/"


def test_basics(account, region, role, s3_bucket, s3_location, image_list):
    assert len(image_list) > 0, "Unable to find images for testing"
    os.system(f"aws ecr get-login-password --region {region} | docker login --username AWS"
              f" --password-stdin {account}.dkr.ecr.{region}.amazonaws.com")
    upload_test_script_to_s3(s3_bucket, s3_location)
    for image_path in image_list:
        single_image_test(account, role, s3_bucket, s3_location, image_path)


def upload_test_script_to_s3(s3_bucket, s3_location):
    credentials = get_session().get_credentials()
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=credentials.access_key,
        aws_secret_access_key=credentials.secret_key,
        aws_session_token=credentials.token,
    )
    s3_client.upload_file(SCRIPT_PATH + SCRIPT_NAME, s3_bucket, f"{s3_location}/{SCRIPT_NAME}")


def single_image_test(account, role, s3_bucket, s3_location, image_path):
    environment_variables = {
        "AMZN_BRAKET_SCRIPT_S3_URI": f"s3://{s3_bucket}/{s3_location}/{SCRIPT_NAME}",
        "AMZN_BRAKET_SCRIPT_ENTRY_POINT": f"{SCRIPT_NAME}",
    }
    estimator = Estimator(image_uri=image_path,
                          role=f"arn:aws:iam::{account}:role/{role}",
                          instance_count=1,
                          instance_type='local',
                          hyperparameters=environment_variables,
                          environment=environment_variables)
    estimator_output = io.StringIO()
    with redirect_stdout(estimator_output):
        try:
            estimator.fit()
        except Exception as e:
            print(e)
    output = estimator_output.getvalue()
    print(output)
    assert output.find("Braket Container Run Success") > 0, "Container did not run successfully"
    assert output.find("exited with code 0") > 0, "Exit code was not zero"
