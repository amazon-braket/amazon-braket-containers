import json
import re
import tempfile
from pathlib import Path
from unittest import mock
from urllib.parse import urlparse

import pytest

from src.braket_container import (
    create_paths,
    create_symlink,
    download_customer_code,
    log_failure_and_exit,
    unpack_code_and_add_to_path,
    get_code_setup_parameters,
    setup_and_run,
    try_bind_hyperparameters_to_customer_method,
    install_additional_requirements,
    run_customer_code,
)


@mock.patch('pathlib._normal_accessor.mkdir')
@mock.patch('src.braket_container.sys')
def test_log_failure_logging(mock_sys, mock_mkdir):
    with mock.patch('builtins.open', mock.mock_open()) as file_open:
        log_failure_and_exit("my test data")
        # Open with append in case someone (eg. the customer) wrote something to the file already
        file_open.assert_called_with('/opt/ml/output/failure', 'a')
        file_write = file_open()
        file_write.write.assert_called_with("my test data")
    # We use the /opt/ml/output directory in case there is an error during symlink
    mock_mkdir.assert_called_with(Path('/opt/ml/output'), 511)
    mock_sys.exit.assert_called_with(0)


@mock.patch('src.braket_container.os')
def test_create_symlink(mock_os):
    create_symlink()
    mock_os.symlink.assert_called_with('/opt/ml', '/opt/braket')


@pytest.mark.xfail(raises=PermissionError)
@mock.patch('src.braket_container.log_failure_and_exit')
@mock.patch('src.braket_container.os.symlink')
def test_create_symlink_error(mock_symlink, mock_log_failure):
    mock_symlink.side_effect = PermissionError
    create_symlink()
    mock_log_failure.assert_called()


@mock.patch('pathlib._normal_accessor.mkdir')
def test_create_paths(mock_mkdir):
    create_paths()
    mock_mkdir.assert_any_call(Path('/opt/braket/code/customer_code'), 511)
    mock_mkdir.assert_any_call(Path('/opt/braket/code/customer_code/original'), 511)
    mock_mkdir.assert_any_call(Path('/opt/braket/code/customer_code/extracted'), 511)
    mock_mkdir.assert_any_call(Path('/opt/braket/additional_setup'), 511)
    assert mock_mkdir.call_count == 4


@mock.patch('src.braket_container.boto3')
def test_download_customer_code(mock_boto):
    mock_s3 = mock_boto.client.return_value = mock.MagicMock()
    result_file = download_customer_code('file://test_s3_bucket/test_s3_loc')
    mock_s3.download_file.assert_called_with('test_s3_bucket', 'test_s3_loc',
                                             '/opt/braket/code/customer_code/original/test_s3_loc')
    assert result_file == '/opt/braket/code/customer_code/original/test_s3_loc'


@mock.patch('src.braket_container.shutil')
def test_unpack_code_and_add_to_path_non_zipped(mock_shutil):
    file_path = urlparse('file://test_s3_bucket/test_s3_loc')
    unpack_code_and_add_to_path(file_path, "")
    mock_shutil.copy.assert_called_with(file_path, '/opt/braket/code/customer_code/extracted')


@pytest.mark.parametrize(
    "compression_type", ["gzip", "zip", "Gzip", " Gzip", " GZIP "]
)
@mock.patch('src.braket_container.shutil')
def test_unpack_code_and_add_to_path_zipped(mock_shutil, compression_type):
    file_path = urlparse('file://test_s3_bucket/test_s3_loc')
    unpack_code_and_add_to_path(file_path, compression_type)
    mock_shutil.unpack_archive.assert_called_with(file_path,
                                                  '/opt/braket/code/customer_code/extracted')


@pytest.mark.parametrize(
    "environment", [
        {
            "expected_result" : None
        },
        {
            "set_vars" : {},
            "expected_result" : None
        },
        {
            "set_vars": {
                "AMZN_BRAKET_SCRIPT_S3_URI" : "test_s3_uri",
                "AMZN_BRAKET_SCRIPT_ENTRY_POINT" : "test_entry_point",
                "AMZN_BRAKET_SCRIPT_COMPRESSION_TYPE" : "test_comp"
            },
            "expected_result": ["test_s3_uri", "test_entry_point", "test_comp"]
        },
        {
            "set_vars": {
                "AMZN_BRAKET_SCRIPT_S3_URI" : "test_s3_uri",
                "AMZN_BRAKET_SCRIPT_ENTRY_POINT" : "test_entry_point",
            },
            "expected_result": ["test_s3_uri", "test_entry_point", None]
        },
        {
            "set_vars": {
                "SM_HPS": "",
            },
            "expected_result": None
        },
        {
            "set_vars": {
                "SM_HPS": "invalid json",
            },
            "expected_result": None
        },
        {
            "set_vars": {
                "SM_HPS": "{\"AMZN_BRAKET_SCRIPT_S3_URI\":\"test_s3_uri\", \"AMZN_BRAKET_SCRIPT_ENTRY_POINT\":\"test_entry_point\", \"AMZN_BRAKET_SCRIPT_COMPRESSION_TYPE\":\"test_comp\"}",
            },
            "expected_result": ["test_s3_uri", "test_entry_point", "test_comp"]
        },
        {
            "set_vars": {
                "SM_HPS": "{\"AMZN_BRAKET_SCRIPT_S3_URI\":\"test_s3_uri\", \"AMZN_BRAKET_SCRIPT_ENTRY_POINT\":\"test_entry_point\"}",
            },
            "expected_result": ["test_s3_uri", "test_entry_point", None]
        },
    ]
)
@mock.patch('src.braket_container.log_failure_and_exit')
@mock.patch('src.braket_container.sys')
def test_get_code_setup_parameters(mock_sys, mock_log_failure, environment, monkeypatch):
    set_vars = environment.setdefault("set_vars", {})
    for key in set_vars:
        monkeypatch.setenv(key, set_vars[key])
    s3_uri, entry_point, compression_type = get_code_setup_parameters()
    expected = environment["expected_result"]
    if expected:
        mock_log_failure.assert_not_called()
        assert s3_uri == expected[0]
        assert entry_point == expected[1]
        assert compression_type == expected[2]
    else:
        mock_log_failure.assert_called()


@pytest.mark.parametrize(
    "file_walk_results", [
        [("my_root", [], ["requirements.txt"])],
        [("my_root", [], ["devrequirements.txt", "requirements.txt", "requirements.txt.bak"])],
        [("empty_root", [], []), ("my_root", [], ["requirements.txt"])],
        [("my_root", [], ["requirements.txt"]), ("my_root", [], ["devrequirements.txt"])],
    ]
)
@mock.patch('src.braket_container.subprocess')
@mock.patch('src.braket_container.os')
def test_install_additional_requirements(mock_os, mock_subprocess, file_walk_results):
    mock_os.walk.return_value = file_walk_results
    mock_os.path.join.return_value = "joined_path"
    install_additional_requirements()
    mock_os.path.join.assert_called_with("my_root", "requirements.txt")
    mock_subprocess.run.assert_called_with(
        ["python", "-m", "pip", "install", "-r", "joined_path"],
        cwd='/opt/braket/code/customer_code/extracted',
    )
    assert mock_subprocess.run.call_count == 1


def customer_function():
    print("Hello")
    return 0


@mock.patch("src.braket_container._log_failure")
@mock.patch('src.braket_container.importlib')
@mock.patch('src.braket_container.get_code_setup_parameters')
@mock.patch('src.braket_container.shutil')
@mock.patch('src.braket_container.boto3')
@mock.patch('pathlib._normal_accessor.mkdir')
@mock.patch('src.braket_container.os')
@mock.patch('src.braket_container.sys')
def test_run_customer_code_function(
    mock_sys,
    mock_os,
    mock_mkdir,
    mock_boto,
    mock_shutil,
    mock_get_code_setup,
    mock_importlib,
    mock_log_failure,
    hyperparameters_json
):
    mock_os.getenv = lambda x: (
        "hyperparameters.json"
        if x == "AMZN_BRAKET_HP_FILE"
        else ""
    )
    mock_get_code_setup.return_value = (
        "s3://test_bucket/test_location",
        "test_module:customer_function",
        None,
    )
    mock_importlib.import_module.return_value.customer_function = customer_function

    run_customer_code()


def customer_function_fails():
    open("fake_file")


@mock.patch('src.braket_container.importlib')
@mock.patch('src.braket_container.get_code_setup_parameters')
@mock.patch('src.braket_container.shutil')
@mock.patch('src.braket_container.boto3')
@mock.patch('pathlib._normal_accessor.mkdir')
@mock.patch('src.braket_container.os.getenv')
@mock.patch('src.braket_container.sys')
def test_run_customer_code_function_fails(
    mock_sys,
    mock_getenv,
    mock_mkdir,
    mock_boto,
    mock_shutil,
    mock_get_code_setup,
    mock_importlib,
    hyperparameters_json,
):
    mock_getenv.side_effect = lambda x, y = None: (
        "hyperparameters.json"
        if x == "AMZN_BRAKET_HP_FILE"
        else y or ""
    )
    mock_get_code_setup.return_value = (
        "s3://test_bucket/test_location",
        "test_module:customer_function_fails",
        None,
    )
    mock_importlib.import_module.return_value.customer_function_fails = customer_function_fails

    with tempfile.TemporaryDirectory() as tempdir:
        import src
        extracted_code_path = src.braket_container.EXTRACTED_CUSTOMER_CODE_PATH
        log_file_name = src.braket_container.ERROR_LOG_FILE
        mock_log_file_name = Path(tempdir, "failure")
        try:
            src.braket_container.EXTRACTED_CUSTOMER_CODE_PATH = tempdir
            src.braket_container.ERROR_LOG_FILE = mock_log_file_name

            run_customer_code()

            mock_sys.exit.assert_called_with(1)
            with open(mock_log_file_name, "r") as f:
                assert f.read() == "FileNotFoundError: [Errno 2] No such file or directory: 'fake_file'"
        finally:
            src.braket_container.EXTRACTED_CUSTOMER_CODE_PATH = extracted_code_path
            src.braket_container.ERROR_LOG_FILE = log_file_name



def customer_method_no_args():
    return


def customer_method_no_annotations(some_float_arg, some_string_arg):
    return


def customer_method_match(some_float_arg: float, some_string_arg: str):
    return


def customer_method_flipped(some_string_arg, some_float_arg):
    return

def customer_method_defaults(
    some_float_arg: float = 0.1,
    some_string_arg: str = "",
    some_other_arg=None,
):
    return


def customer_method_wrong_type(some_float_arg: int, some_string_arg: str):
    return


@pytest.fixture
def hyperparameters(pytester):
    # these are already converted to strings by sagemaker
    hp_map = {
        "no_hps": {},
        "hps": {
            "some_float_arg": "3.14",
            "some_string_arg": "my_string",
        },
    }
    pytester.makefile(
        ".json",
        no_hps=json.dumps(hp_map["no_hps"])
    )
    pytester.makefile(
        ".json",
        hps=json.dumps(hp_map["hps"])
    )


@pytest.mark.parametrize(
    "hp_file, customer_method",
    (
        ("no_hps.json", customer_method_no_args),
        ("no_hps.json", customer_method_defaults),
        ("hps.json", customer_method_no_annotations),
        ("hps.json", customer_method_match),
        ("hps.json", customer_method_flipped),
        ("hps.json", customer_method_defaults),
    )
)
def test_bind_hyperparameters_successful(customer_method, hp_file, hyperparameters):
    with mock.patch.dict("os.environ", {"AMZN_BRAKET_HP_FILE": hp_file}):
        binding = try_bind_hyperparameters_to_customer_method(customer_method)
    customer_method(**binding)


@pytest.mark.parametrize(
    "hp_file, customer_method",
    (
        ("no_hps.json", customer_method_no_annotations),
        ("no_hps.json", customer_method_match),
        ("no_hps.json", customer_method_flipped),
        ("hps.json", customer_method_no_args),
    )
)
def test_bind_hyperparameters_skipped(customer_method, hp_file, hyperparameters):
    with mock.patch.dict("os.environ", {"AMZN_BRAKET_HP_FILE": hp_file}):
        binding = try_bind_hyperparameters_to_customer_method(customer_method)
    assert binding is None


def test_bind_hyperparameters_type_error(hyperparameters):
    hp_file = "hps.json"
    invalid_literal = re.escape("invalid literal for int() with base 10: '3.14'")
    with mock.patch.dict("os.environ", {"AMZN_BRAKET_HP_FILE": hp_file}):
        with pytest.raises(ValueError, match=invalid_literal):
            try_bind_hyperparameters_to_customer_method(customer_method_wrong_type)
