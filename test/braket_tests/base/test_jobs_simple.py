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

from ..common.braket_jobs_util import job_test


def test_bell_circuit(account, region, role, s3_bucket, s3_location, image_list):
    assert len(image_list) > 0, "Unable to find images for testing"
    create_job_args = {
        "source_module": "./test/resources/bell_circuit.py",
    }
    for image_path in image_list:
        job_test(account, role, s3_bucket, image_path, "bell", **create_job_args)
