from pathlib import Path
from unittest import mock
from urllib.parse import urlparse

import pytest

from src.braket_container import (
    create_paths,
    create_symlink,
    install_additional_libraries,
    download_customer_code,
    log_failure,
    unpack_code_and_add_to_path,
    get_code_setup_parameters,
    setup_and_run
)

@mock.patch('pathlib._normal_accessor.mkdir')
def test_log_failure_logging(mock_mkdir):
    with mock.patch('builtins.open', mock.mock_open()) as file_open:
        log_failure("my test data")
        # Open with append in case someone (eg. the customer) wrote something to the file already
        file_open.assert_called_with('/opt/ml/output/failure', 'a')
        file_write = file_open()
        file_write.write.assert_called_with("my test data")
    # We use the /opt/ml/output directory in case there is an error during symlink
    mock_mkdir.assert_called_with(Path('/opt/ml/output'), 511)


@mock.patch('src.braket_container.os')
def test_create_symlink(mock_os):
    create_symlink()
    mock_os.symlink.assert_called_with('/opt/ml', '/opt/braket')


@pytest.mark.xfail(raises=PermissionError)
@mock.patch('src.braket_container.log_failure')
@mock.patch('src.braket_container.os.symlink')
def test_create_symlink_error(mock_symlink, mock_log_failure):
    mock_symlink.side_effect = PermissionError
    create_symlink()


@mock.patch('pathlib._normal_accessor.mkdir')
def test_create_paths(mock_mkdir):
    create_paths()
    mock_mkdir.assert_any_call(Path('/opt/braket/code/customer_code'), 511)
    mock_mkdir.assert_any_call(Path('/opt/braket/code/customer_code/original'), 511)
    mock_mkdir.assert_any_call(Path('/opt/braket/code/customer_code/extracted'), 511)
    mock_mkdir.assert_any_call(Path('/opt/braket/additional_lib/original'), 511)
    mock_mkdir.assert_any_call(Path('/opt/braket/additional_lib/extracted'), 511)
    assert mock_mkdir.call_count == 5


@mock.patch('src.braket_container.subprocess')
@mock.patch('src.braket_container.os.scandir')
@mock.patch('src.braket_container.shutil')
@mock.patch('src.braket_container.boto3')
def test_install_additional_libraries(mock_boto, mock_shutil, mock_scandir, subprocess, monkeypatch):
    mock_s3 = mock_boto.client.return_value = mock.MagicMock()
    item0 = mock.MagicMock()
    item0.is_dir.return_value = False
    item1 = mock.MagicMock()
    item1.is_dir.return_value = True
    item1.path = "temp_path"
    mock_scandir.return_value = [item0, item1]
    monkeypatch.setenv("AMZN_BRAKET_IMAGE_ADDITIONAL_LIB", "s3://test_bucket/test_location")
    install_additional_libraries()
    mock_s3.download_file.assert_called_with("test_bucket", "test_location",
                                             "/opt/braket/additional_lib/original/test_location")
    mock_shutil.unpack_archive.assert_called_with(
        "/opt/braket/additional_lib/original/test_location",
        "/opt/braket/additional_lib/extracted"
    )
    assert subprocess.run.call_count == 1
    subprocess.run.assert_called_with(["python", "-m", "pip", "install", "temp_path"])


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
    mock_shutil.move.assert_called_with(file_path, '/opt/braket/code/customer_code/extracted')


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
@mock.patch('src.braket_container.log_failure')
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
        mock_sys.exit.assert_called_with(1)


@mock.patch('src.braket_container.subprocess')
@mock.patch('src.braket_container.get_code_setup_parameters')
@mock.patch('src.braket_container.shutil')
@mock.patch('src.braket_container.boto3')
@mock.patch('pathlib._normal_accessor.mkdir')
@mock.patch('src.braket_container.os')
@mock.patch('src.braket_container.sys')
def test_setup_and_run_as_subprocess(mock_sys, mock_os, mock_mkdir, mock_boto, mock_shutil, mock_get_code_setup, mock_subprocess):
    # Setup
    expected_return_value = 3
    mock_os.getenv.return_value = ""
    mock_get_code_setup.return_value = "s3://test_bucket/test_location", "test_entry_point", None
    run_result_object = mock.MagicMock()
    run_result_object.returncode = expected_return_value
    mock_subprocess.run.return_value = run_result_object

    # Act
    setup_and_run()

    # Assert
    mock_subprocess.run.assert_called_with(
        ["python", "-m", "test_entry_point"],
    )
    mock_sys.exit.assert_called_with(expected_return_value)


@mock.patch('src.braket_container.multiprocessing')
@mock.patch('src.braket_container.importlib')
@mock.patch('src.braket_container.get_code_setup_parameters')
@mock.patch('src.braket_container.shutil')
@mock.patch('src.braket_container.boto3')
@mock.patch('pathlib._normal_accessor.mkdir')
@mock.patch('src.braket_container.os')
@mock.patch('src.braket_container.sys')
def test_setup_and_run_as_process(mock_sys, mock_os, mock_mkdir, mock_boto, mock_shutil, mock_get_code_setup, mock_importlib, mock_process):
    # Setup
    expected_return_value = 3
    mock_os.getenv.return_value = ""
    mock_get_code_setup.return_value = "s3://test_bucket/test_location", "test_module:test_function", None
    mock_process_object = mock.MagicMock()
    mock_process.Process.return_value = mock_process_object
    mock_process_object.exitcode = expected_return_value

    # Act
    setup_and_run()

    # Assert
    mock_process_object.start.assert_called()
    mock_process_object.join.assert_called()
    mock_sys.exit.assert_called_with(expected_return_value)
