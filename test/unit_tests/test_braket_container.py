import importlib
import json
import os
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
    wrap_customer_code,
    EXTRACTED_CUSTOMER_CODE_PATH,
)


@mock.patch("pathlib.Path.mkdir")
@mock.patch("src.braket_container.sys")
def test_log_failure_logging(mock_sys, mock_mkdir):
    with mock.patch("builtins.open", mock.mock_open()) as file_open:
        log_failure_and_exit("my test data")
        # Open with append in case someone (eg. the customer) wrote something to the file already
        file_open.assert_called_with("/opt/ml/output/failure", "a")
        file_write = file_open()
        file_write.write.assert_called_with("my test data")
    # We use the /opt/ml/output directory in case there is an error during symlink
    mock_mkdir.assert_called_with(parents=True, exist_ok=True)
    mock_sys.exit.assert_called_with(0)


@mock.patch("src.braket_container.os")
def test_create_symlink(mock_os):
    create_symlink()
    mock_os.symlink.assert_called_with("/opt/ml", "/opt/braket")


@pytest.mark.xfail(raises=PermissionError)
@mock.patch("src.braket_container.log_failure_and_exit")
@mock.patch("src.braket_container.os.symlink")
def test_create_symlink_error(mock_symlink, mock_log_failure):
    mock_symlink.side_effect = PermissionError
    create_symlink()
    mock_log_failure.assert_called()


@mock.patch("pathlib.Path.mkdir")
def test_create_paths(mock_mkdir):
    create_paths()
    mock_mkdir.assert_called_with(parents=True, exist_ok=True)
    assert mock_mkdir.call_count == 4


@mock.patch("src.braket_container.boto3")
def test_download_customer_code(mock_boto):
    mock_s3 = mock_boto.client.return_value = mock.MagicMock()
    result_file = download_customer_code("file://test_s3_bucket/test_s3_loc")
    mock_s3.download_file.assert_called_with("test_s3_bucket", "test_s3_loc",
                                             "/opt/braket/code/customer_code/original/test_s3_loc")
    assert result_file == "/opt/braket/code/customer_code/original/test_s3_loc"


@mock.patch("src.braket_container.shutil")
def test_unpack_code_and_add_to_path_non_zipped(mock_shutil):
    file_path = urlparse("file://test_s3_bucket/test_s3_loc")
    unpack_code_and_add_to_path(file_path, "")
    mock_shutil.copy.assert_called_with(file_path, "/opt/braket/code/customer_code/extracted")


@pytest.mark.parametrize(
    "compression_type", ["gzip", "zip", "Gzip", " Gzip", " GZIP "]
)
@mock.patch("src.braket_container.shutil")
def test_unpack_code_and_add_to_path_zipped(mock_shutil, compression_type):
    file_path = urlparse("file://test_s3_bucket/test_s3_loc")
    unpack_code_and_add_to_path(file_path, compression_type)
    mock_shutil.unpack_archive.assert_called_with(file_path,
                                                  "/opt/braket/code/customer_code/extracted")


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
@mock.patch("src.braket_container.log_failure_and_exit")
@mock.patch("src.braket_container.sys")
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
@mock.patch("src.braket_container.subprocess")
@mock.patch("src.braket_container.os")
def test_install_additional_requirements(mock_os, mock_subprocess, file_walk_results):
    mock_os.walk.return_value = file_walk_results
    mock_os.path.join.return_value = "joined_path"
    install_additional_requirements()
    mock_os.path.join.assert_called_with("my_root", "requirements.txt")
    mock_subprocess.run.assert_called_with(
        ["python", "-m", "pip", "install", "-r", "joined_path"],
        cwd="/opt/braket/code/customer_code/extracted",
    )
    assert mock_subprocess.run.call_count == 1


def customer_function():
    print("Hello")
    return 0


@mock.patch("src.braket_container.multiprocessing")
@mock.patch("src.braket_container.importlib")
@mock.patch("src.braket_container.get_code_setup_parameters")
@mock.patch("src.braket_container.shutil")
@mock.patch("src.braket_container.boto3")
@mock.patch("pathlib.Path.mkdir")
@mock.patch("src.braket_container.os.getenv")
@mock.patch("src.braket_container.sys")
def test_run_customer_code_function(
    mock_sys,
    mock_getenv,
    mock_mkdir,
    mock_boto,
    mock_shutil,
    mock_get_code_setup,
    mock_importlib,
    mock_mp,
    hyperparameters_json,
):
    mock_getenv.side_effect = lambda x, y = None: (
        "hyperparameters.json"
        if x == "AMZN_BRAKET_HP_FILE"
        else y or ""
    )
    mock_get_code_setup.return_value = (
        "s3://test_bucket/test_location",
        "test_module:customer_function",
        None,
    )
    mock_process = mock.MagicMock()
    mock_mp.Process.return_value = mock_process

    run_customer_code()

    customer_fn = mock_importlib.import_module.return_value.customer_function
    # Process target must be a picklable module-level function (not a closure),
    # because cudaq imports switch multiprocessing to forkserver which pickles
    # the target.
    mock_mp.Process.assert_called_with(
        target=wrap_customer_code,
        args=(customer_fn,),
        kwargs={},
    )
    mock_process.start.assert_called_with()
    mock_process.join.assert_called_with()


def customer_function_fails():
    open("fake_file")


def test_wrap_customer_code_is_module_level_picklable():
    """Regression test for a pre-existing bug where `wrap_customer_code` was a
    closure factory returning a nested function, which `multiprocessing.Process`
    could not pickle under the `forkserver` start method (imposed by some user
    imports, notably cudaq).

    The fix was to make `wrap_customer_code` a module-level function taking the
    customer_method as an argument. This test guards that invariant by pickling
    it directly.
    """
    import pickle

    # Module-level functions pickle cleanly.
    data = pickle.dumps(wrap_customer_code)
    restored = pickle.loads(data)
    assert restored is wrap_customer_code


@mock.patch("src.braket_container._log_failure")
@mock.patch("os.chdir")
def test_wrap_customer_code_logs_failure(mock_cd, mock_log):
    file_not_found = re.escape("[Errno 2] No such file or directory: 'fake_file'")
    with pytest.raises(FileNotFoundError, match=file_not_found):
        wrap_customer_code(customer_function_fails)

    # Check that chdir was called with the extracted code path and back
    assert mock_cd.call_count >= 1
    mock_log.assert_called_with(
        "FileNotFoundError: [Errno 2] No such file or directory: 'fake_file'",
        display=False,
    )




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


@pytest.fixture
def dp_hp_files(pytester):
    pytester.makefile(
        ".json",
        dp_on=json.dumps({"sagemaker_distributed_dataparallel_enabled": "true"}),
    )
    pytester.makefile(
        ".json",
        dp_off=json.dumps({"sagemaker_distributed_dataparallel_enabled": "false"}),
    )
    pytester.makefile(".json", dp_missing=json.dumps({}))


@pytest.mark.parametrize(
    "hp_file, expected",
    (
        ("dp_on.json", True),
        ("dp_off.json", False),
        ("dp_missing.json", False),
    ),
)
def test_is_data_parallel_enabled(hp_file, expected, dp_hp_files):
    from src.braket_container import _is_data_parallel_enabled

    with mock.patch.dict("os.environ", {"AMZN_BRAKET_HP_FILE": hp_file}):
        assert _is_data_parallel_enabled() is expected


def test_is_data_parallel_enabled_no_hp_file():
    from src.braket_container import _is_data_parallel_enabled

    with mock.patch.dict("os.environ", clear=True):
        assert _is_data_parallel_enabled() is False


def test_data_parallel_topology_single_node():
    from src.braket_container import _data_parallel_topology

    fake_torch = mock.MagicMock()
    fake_torch.cuda.device_count.return_value = 4
    env = {}  # no SM_HOSTS / SM_CURRENT_HOST
    with mock.patch.dict("sys.modules", {"torch": fake_torch}), \
            mock.patch.dict("os.environ", env, clear=True):
        topo = _data_parallel_topology()

    assert topo.nnodes == 1
    assert topo.nproc_per_node == 4
    assert topo.world_size == 4
    assert topo.node_rank == 0
    assert topo.master_addr == "127.0.0.1"
    assert topo.master_port == "23456"
    assert topo.rank_offset == 0


def test_data_parallel_topology_multi_node():
    from src.braket_container import _data_parallel_topology

    fake_torch = mock.MagicMock()
    fake_torch.cuda.device_count.return_value = 4
    env = {
        "SM_HOSTS": json.dumps(["algo-1", "algo-2"]),
        "SM_CURRENT_HOST": "algo-2",
    }
    with mock.patch.dict("sys.modules", {"torch": fake_torch}), \
            mock.patch.dict("os.environ", env, clear=True):
        topo = _data_parallel_topology()

    assert topo.nnodes == 2
    assert topo.nproc_per_node == 4
    assert topo.world_size == 8
    assert topo.node_rank == 1            # algo-2 is the second host once sorted
    assert topo.master_addr == "algo-1"   # smallest host name is the master
    # Worker 0 on this node is global rank 4 (offset 1 * 4).
    assert topo.rank_offset == 4


def test_data_parallel_topology_no_torch():
    from src.braket_container import _data_parallel_topology

    # If torch can't be imported (e.g. unit tests on a machine without it),
    # the launcher gracefully falls back to a single worker.
    with mock.patch.dict("sys.modules", {"torch": None}), \
            mock.patch.dict("os.environ", {}, clear=True):
        topo = _data_parallel_topology()

    assert topo.nproc_per_node == 1
    assert topo.world_size == 1


def test_dp_worker_sets_env_and_invokes_customer():
    from src.braket_container import DataParallelTopology, _dp_worker_target

    topology = DataParallelTopology(
        nnodes=2,
        nproc_per_node=4,
        world_size=8,
        node_rank=1,
        master_addr="algo-1",
        master_port="23456",
    )
    captured = {}

    def fake_wrap_customer_code(customer_code, **kwargs):
        # Snapshot the env so we can assert the worker set things correctly
        # before the customer code was invoked.
        captured["env"] = {
            k: os.environ.get(k)
            for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")
        }
        captured["customer_code"] = customer_code
        captured["kwargs"] = kwargs

    customer = mock.MagicMock(__name__="customer")
    with mock.patch.dict("os.environ", {}, clear=True), \
            mock.patch(
                "src.braket_container.wrap_customer_code",
                side_effect=fake_wrap_customer_code,
            ):
        _dp_worker_target(
            local_rank=2,
            topology=topology,
            customer_code=customer,
            kwargs={"foo": "bar"},
        )

    # node_rank=1 * nproc_per_node=4 + local_rank=2 == global rank 6
    assert captured["env"]["RANK"] == "6"
    assert captured["env"]["LOCAL_RANK"] == "2"
    assert captured["env"]["WORLD_SIZE"] == "8"
    assert captured["env"]["MASTER_ADDR"] == "algo-1"
    assert captured["env"]["MASTER_PORT"] == "23456"
    assert captured["customer_code"] is customer
    assert captured["kwargs"] == {"foo": "bar"}


def test_kick_off_data_parallel_spawns_one_process_per_gpu():
    from src.braket_container import DataParallelTopology, kick_off_data_parallel

    topology = DataParallelTopology(
        nnodes=1, nproc_per_node=3, world_size=3, node_rank=0,
        master_addr="127.0.0.1", master_port="23456",
    )
    customer = mock.MagicMock(__name__="customer")

    fake_proc = mock.MagicMock(exitcode=0)
    with mock.patch("src.braket_container._data_parallel_topology", return_value=topology), \
            mock.patch(
                "src.braket_container.try_bind_hyperparameters_to_customer_method",
                return_value=None,
            ), \
            mock.patch(
                "src.braket_container.multiprocessing.Process", return_value=fake_proc,
            ) as mock_process:
        rc = kick_off_data_parallel(customer)

    assert rc == 0
    assert mock_process.call_count == 3                           # one per GPU
    assert fake_proc.start.call_count == 3
    assert fake_proc.join.call_count == 3
    # Each Process is constructed with local_rank in {0, 1, 2}.
    local_ranks = [c.kwargs["args"][0] for c in mock_process.call_args_list]
    assert sorted(local_ranks) == [0, 1, 2]


def test_kick_off_data_parallel_propagates_worker_failure():
    from src.braket_container import DataParallelTopology, kick_off_data_parallel

    topology = DataParallelTopology(
        nnodes=1, nproc_per_node=2, world_size=2, node_rank=0,
        master_addr="127.0.0.1", master_port="23456",
    )

    failing_proc = mock.MagicMock(exitcode=42)
    with mock.patch("src.braket_container._data_parallel_topology", return_value=topology), \
            mock.patch(
                "src.braket_container.try_bind_hyperparameters_to_customer_method",
                return_value=None,
            ), \
            mock.patch(
                "src.braket_container.multiprocessing.Process",
                return_value=failing_proc,
            ):
        rc = kick_off_data_parallel(mock.MagicMock(__name__="customer"))

    assert rc == 42
