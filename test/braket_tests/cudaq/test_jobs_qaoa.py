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

import time

from ..common.braket_jobs_util import job_test


def test_qaoa_circuit(account, role, s3_bucket, image_list, use_local_jobs, use_local_sim):
    assert len(image_list) > 0, "Unable to find images for testing"

    for image_path in image_list:
        job_test(account, role, s3_bucket, image_path, use_local_jobs, use_local_sim, "cudaq-qaoa", "cudaq")