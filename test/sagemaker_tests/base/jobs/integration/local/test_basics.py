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

from sagemaker.estimator import Estimator


def test_basics(account, region, role, image_list):
    assert len(image_list) > 0, "Unable to find images for testing"
    os.system(f"aws ecr get-login-password --region {region} | docker login --username AWS"
              f" --password-stdin {account}.dkr.ecr.{region}.amazonaws.com")
    for image_path in image_list:
        single_image_test(account, role, image_path)


def single_image_test(account, role, image_path):
    estimator = Estimator(image_uri=image_path,
                          role=f"arn:aws:iam::{account}:role/{role}",
                          instance_count=1,
                          instance_type='local')
    estimator_output = io.StringIO()
    with redirect_stdout(estimator_output):
        estimator.fit()
    output = estimator_output.getvalue()
    assert output.find("Braket Container Run Success") > 0
    assert output.find("exited with code 0") > 0
