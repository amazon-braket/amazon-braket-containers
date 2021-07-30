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

import errno
import importlib
import os
import json
import shutil
import subprocess
import sys
from multiprocessing import Process
from pathlib import Path
from urllib.parse import urlparse

import boto3
from botocore.session import get_session

OPT_ML = os.path.join("/opt", "ml")
OPT_BRAKET = os.path.join("/opt", "braket")
CUSTOMER_CODE_PATH = os.path.join(OPT_BRAKET, "code", "customer_code")
ORIGINAL_CUSTOMER_CODE_PATH = os.path.join(CUSTOMER_CODE_PATH, "original")
EXTRACTED_CUSTOMER_CODE_PATH = os.path.join(CUSTOMER_CODE_PATH, "extracted")
ERROR_LOG_PATH = os.path.join(OPT_ML, "output")
ERROR_LOG_FILE = os.path.join(ERROR_LOG_PATH, "failure")


def log_failure(*args):
    """
    Log failures to a file so that it can be parsed by the backend service and included in
    failure messages for a job.

    Args:
        args: variable list of text to write to the file.
    """
    Path(ERROR_LOG_PATH).mkdir(parents=True, exist_ok=True)
    with open(ERROR_LOG_FILE, 'a') as error_log:
        for text in args:
            error_log.write(text)
            print(text)


def create_paths():
    """
    These paths are created early on so that the rest of the code can assume that the directories
    are available when needed.
    """
    Path(CUSTOMER_CODE_PATH).mkdir(parents=True, exist_ok=True)
    Path(ORIGINAL_CUSTOMER_CODE_PATH).mkdir(parents=True, exist_ok=True)
    Path(EXTRACTED_CUSTOMER_CODE_PATH).mkdir(parents=True, exist_ok=True)


def create_symlink():
    """
    The ML paths are inserted by the backend service by default. To prevent confusion we link
    the Braket paths to it (to unify them), and use the Braket paths from now on.
    """
    try:
        os.symlink(OPT_ML, OPT_BRAKET)
    except OSError as e:
        if e.errno != errno.EEXIST:
            print(f"Got unexpected exception: {e}")
            log_failure("Symlink failure")
            raise e


def download_customer_code(s3_client, s3_uri):
    """
    Downloads the customer code to the original customer path. The code is assumed to be a single
    file in S3. The file may be a compressed archive containing all the customer code.

    Args:
        s3_client: the S3 client.
        s3_uri: the S3 URI to get the code from.
    Returns:
        the path to the file containing the code.
    """
    s3_bucket = s3_uri.netloc
    s3_key = s3_uri.path.lstrip("/")
    local_s3_file = os.path.join(ORIGINAL_CUSTOMER_CODE_PATH, os.path.basename(s3_key))
    s3_client.download_file(s3_bucket, s3_key, local_s3_file)
    return local_s3_file


def unpack_code_and_add_to_path(local_s3_file, compression_type):
    """
    Unpack the customer code, if necessary. Add the customer code to the system path.

    Args:
        local_s3_file: the file representing the customer code.
        compression_type: if the customer code is stored in an archive, this value will
            represent the compression type of the archive.
    """
    if compression_type in ["gzip", "zip"]:
        try:
            shutil.unpack_archive(local_s3_file, EXTRACTED_CUSTOMER_CODE_PATH)
        except Exception as e:
            log_failure(
                f"Got an exception while trying to unpack archive: {local_s3_file} of type: "
                f"{compression_type}.\nException: {e}"
            )
            sys.exit(1)
    else:
        shutil.move(local_s3_file, EXTRACTED_CUSTOMER_CODE_PATH)
    sys.path.append(EXTRACTED_CUSTOMER_CODE_PATH)


def kick_off_customer_script(entry_point):
    """
    Runs the customer script as a separate process.

    Args:
        entry_point: the entry point to the customer code, represented as <module>:<method>.
    """
    try:
        str_module, _, str_method = entry_point.partition(":")
        customer_module = importlib.import_module(str_module)
        customer_method = getattr(customer_module, str_method)
        customer_code_process = Process(target=customer_method)
        customer_code_process.start()
    except Exception as e:
        log_failure(f"Unable to run job at entry point {entry_point}\nException: {e}")
        sys.exit(1)
    return customer_code_process


def join_customer_script(customer_code_process):
    """
    Joins the process running the customer code.

    Args:
        customer_code_process: the process running the customer code.
    """
    try:
        customer_code_process.join()
    except Exception as e:
        log_failure(f"Job did not exit gracefully.\nException: {e}")
        sys.exit(1)


def get_code_setup_parameters():
    """
    Returns the code setup parameters:
        s3_uri: the S3 location where the code is stored.
        entry_point: the entrypoint into the code.
        compression_type: the compression used to archive the code (optional)
    These values are stored in environment variables, however, we also allow the storing of
    these values in the hyperparameters to facilitate testing in local mode.
    If the s3_uri or entry_point can not be found, the script will exit with an error.

    Returns:
        the code setup parameters as described above.
    """
    s3_uri = os.getenv('AMZN_BRAKET_S3_URI')
    entry_point = os.getenv('AMZN_BRAKET_ENTRY_POINT')
    compression_type = os.getenv('AMZN_BRAKET_COMPRESSION_TYPE')
    if s3_uri and entry_point:
        return s3_uri, entry_point, compression_type
    hyperparameters_env = os.getenv('SM_HPS')
    if hyperparameters_env:
        hyperparameters = json.loads(hyperparameters_env)
        if not s3_uri:
            s3_uri = hyperparameters.get("AMZN_BRAKET_S3_URI")
        if not entry_point:
            entry_point = hyperparameters.get("AMZN_BRAKET_ENTRY_POINT")
        if not compression_type:
            compression_type = hyperparameters.get("AMZN_BRAKET_COMPRESSION_TYPE")
    if not s3_uri:
        log_failure("No customer script specified")
        sys.exit(1)
    if not entry_point:
        log_failure("No customer entry point specified")
        sys.exit(1)
    return s3_uri, entry_point, compression_type


def run_customer_code_as_process(entry_point):
    """
    When provided the name of the package and the method to run, we run them as a process.

    Args:
        entry_point: the code to run in the format <package>:<method>.

    Returns:
        The exit code of the customer code run.
    """
    print("Running Code As Process")
    customer_code_process = kick_off_customer_script(entry_point)
    join_customer_script(customer_code_process)
    print("Code Run Finished")
    return customer_code_process.exitcode


def run_customer_code_as_subprocess(entry_point):
    """
    When provided just the name of the file to run, we run it as a subprocess. This will
    run the subprocess in the directory where the files are extracted.

    Args:
        entry_point: the name of the file to run.

    Returns:
        The exit code of the customer code run.
    """
    print("Running Code As Subprocess")
    result = subprocess.run("python " + entry_point, cwd=EXTRACTED_CUSTOMER_CODE_PATH, shell=True)
    print("Code Run Finished")
    return_code = result.returncode
    return return_code


def run_customer_code(s3_client):
    """
    Downloads and runs the customer code.

    Args:
        s3_client: the S3 client that can be used to download the customer code.

    Returns:
        The exit code of the customer code run.
    """
    print("Initiating Code Setup")
    s3_uri, entry_point, compression_type = get_code_setup_parameters()
    local_s3_file = download_customer_code(s3_client, urlparse(s3_uri, allow_fragments=False))
    unpack_code_and_add_to_path(local_s3_file, compression_type)
    if entry_point.find(":") >= 0:
        return run_customer_code_as_process(entry_point)
    return run_customer_code_as_subprocess(entry_point)


def setup_and_run():
    """
    This method sets up the Braket container, then downloads and runs the customer code.
    """
    print("Beginning Setup")
    credentials = get_session().get_credentials()
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=credentials.access_key,
        aws_secret_access_key=credentials.secret_key,
        aws_session_token=credentials.token,
    )
    create_symlink()
    create_paths()
    exit_code = run_customer_code(s3_client)
    sys.exit(exit_code)


if __name__ == "__main__":
    setup_and_run()
