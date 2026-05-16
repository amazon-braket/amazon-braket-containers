"""CPU-only multi-process integration tests for the data_parallel launcher."""
import multiprocessing
import os
import socket

import pytest

torch = pytest.importorskip("torch")
import torch.distributed as dist  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: E402


def _pick_free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _ddp_smoke_worker():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(backend="gloo", init_method="env://")

    torch.manual_seed(0)
    ddp_model = DDP(nn.Linear(8, 1))
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.1)

    for _ in range(3):
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(ddp_model(torch.randn(2, 8)), torch.randn(2, 1))
        loss.backward()
        optimizer.step()
        if rank == 0:
            import time
            time.sleep(0.05)
        dist.barrier()

    iters = torch.tensor([3], dtype=torch.long)
    dist.all_reduce(iters, op=dist.ReduceOp.SUM)
    assert int(iters.item()) == 3 * world_size

    dist.destroy_process_group()


def _ddp_raises_worker():
    dist.init_process_group(backend="gloo", init_method="env://")
    raise RuntimeError("simulated failure for test")


@pytest.fixture
def fork_start_method():
    prev = multiprocessing.get_start_method(allow_none=True)
    multiprocessing.set_start_method("fork", force=True)
    yield
    if prev is not None:
        multiprocessing.set_start_method(prev, force=True)


def _patch_topology_and_paths(monkeypatch, tmp_path, *, nproc_per_node=2):
    from src import braket_container
    monkeypatch.setattr(braket_container, "EXTRACTED_CUSTOMER_CODE_PATH", str(tmp_path))
    fake_topology = braket_container.DataParallelTopology(
        nnodes=1,
        nproc_per_node=nproc_per_node,
        world_size=nproc_per_node,
        node_rank=0,
        master_addr="127.0.0.1",
        master_port=str(_pick_free_port()),
    )
    monkeypatch.setattr(
        braket_container, "_data_parallel_topology", lambda: fake_topology
    )
    return braket_container


def test_kick_off_data_parallel_cpu_smoke(monkeypatch, tmp_path, fork_start_method):
    bc = _patch_topology_and_paths(monkeypatch, tmp_path, nproc_per_node=2)
    assert bc.kick_off_data_parallel(_ddp_smoke_worker) == 0


def test_kick_off_data_parallel_propagates_customer_failure(
    monkeypatch, tmp_path, fork_start_method
):
    bc = _patch_topology_and_paths(monkeypatch, tmp_path, nproc_per_node=2)
    assert bc.kick_off_data_parallel(_ddp_raises_worker) != 0
