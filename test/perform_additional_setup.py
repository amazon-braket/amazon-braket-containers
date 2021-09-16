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
import shutil
import subprocess
import traceback
from urllib.parse import urlparse

import boto3
import tempfile


def download_s3_file(s3_uri: str, local_path: str) -> str:
    """
    Downloads a file to a local path.

    Args:
        s3_uri (str): the S3 URI to get the file from.
        local_path (str) : the local path to download to
    Returns:
        str: the path to the file containing the downloaded path.
    """
    s3_client = boto3.client("s3")
    parsed_url = urlparse(s3_uri, allow_fragments=False)
    s3_bucket = parsed_url.netloc
    s3_key = parsed_url.path.lstrip("/")
    local_s3_file = os.path.join(local_path, os.path.basename(s3_key))
    s3_client.download_file(s3_bucket, s3_key, local_s3_file)
    return local_s3_file


def perform_additional_setup() -> None:
    lib_s3_uri = os.getenv('AMZN_BRAKET_IMAGE_SETUP_SCRIPT')
    if lib_s3_uri:
        try:
            print("Getting setup script from ", lib_s3_uri)
            with tempfile.TemporaryDirectory() as temp_dir:
                script_to_run = download_s3_file(lib_s3_uri, temp_dir)
                subprocess.run(["chmod", "+x", script_to_run])
                subprocess.run(script_to_run)
        except Exception as e:
            print(f"Unable to install additional libraries.\nException: {e}")


if __name__ == "__main__":
    perform_additional_setup()
