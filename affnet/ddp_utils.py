# --------------------------------------------------------
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License
# --------------------------------------------------------


import socket
import torch
import torch.distributed as dist
from typing import Optional
import time

def is_master(opts) -> bool:
    node_rank = getattr(opts, "ddp.rank", 0)
    return node_rank == 0


def dist_barrier():
    dist.barrier()


def dist_monitored_barrier(
    timeout: Optional[float] = None,
    wait_all_ranks: Optional[bool] = False,
    group: Optional = None,
):
    dist.monitored_barrier(group=group, timeout=timeout, wait_all_ranks=wait_all_ranks)


def is_start_rank_node(opts) -> bool:
    node_rank = getattr(opts, "ddp.rank", 0)
    def_rank = getattr(opts, "ddp.start_rank", 0)
    return node_rank == def_rank


def get_world_size():
    return dist.get_world_size()


def get_node_rank():
    return dist.get_rank()


def get_curr_time_stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


text_colors = {
    "logs": "\033[34m",  # 033 is the escape code and 34 is the color code
    "info": "\033[32m",
    "warning": "\033[33m",
    "debug": "\033[93m",
    "error": "\033[31m",
    "bold": "\033[1m",
    "end_color": "\033[0m",
    "light_red": "\033[36m",
}

def warning(message: str) -> None:
    time_stamp = get_curr_time_stamp()
    warn_str = (
        text_colors["warning"]
        + text_colors["bold"]
        + "WARNING"
        + text_colors["end_color"]
    )
    print("{} - {} - {}".format(time_stamp, warn_str, message))


def double_dash_line(dashes: Optional[int] = 75) -> None:
    print(text_colors["error"] + "=" * dashes + text_colors["end_color"])


def info(message: str, print_line: Optional[bool] = False) -> None:
    time_stamp = get_curr_time_stamp()
    info_str = (
        text_colors["info"] + text_colors["bold"] + "INFO   " + text_colors["end_color"]
    )
    print("{} - {} - {}".format(time_stamp, info_str, message))
    if print_line:
        double_dash_line(dashes=150)


def log(message: str, end="\n") -> None:
    time_stamp = get_curr_time_stamp()
    log_str = (
            text_colors["logs"] + text_colors["bold"] + "LOGS   " + text_colors["end_color"]
    )
    print("{} - {} - {}".format(time_stamp, log_str, message), end=end)

def distributed_init(opts) -> int:
    ddp_url = getattr(opts, "ddp.dist_url", None)
    is_master_node = is_master(opts)
    if ddp_url is None:
        ddp_port = getattr(opts, "ddp.dist_port", 6006)
        hostname = socket.gethostname()
        ddp_url = "tcp://{}:{}".format(hostname, ddp_port)
        setattr(opts, "ddp.dist_url", ddp_url)

    node_rank = getattr(opts, "ddp.rank", 0)
    world_size = getattr(opts, "ddp.world_size", 0)
    if torch.distributed.is_initialized():
        warning("DDP is already initialized and cannot be initialize twice!")
    else:
        info("distributed init (rank {}): {}".format(node_rank, ddp_url))

        dist_backend = getattr(opts, "ddp.backend", "nccl")  # "gloo"

        if dist_backend is None and dist.is_nccl_available():
            dist_backend = "nccl"
            if is_master_node:
                log(
                    "Using NCCL as distributed backend with version={}".format(
                        torch.cuda.nccl.version()
                    )
                )
        elif dist_backend is None:
            dist_backend = "gloo"

        dist.init_process_group(
            backend=dist_backend,
            init_method=ddp_url,
            world_size=world_size,
            rank=node_rank,
        )

        # perform a dummy all-reduce to initialize the NCCL communicator
        if torch.cuda.is_available():
            dist.all_reduce(torch.zeros(1).cuda())

    node_rank = torch.distributed.get_rank()
    setattr(opts, "ddp.rank", node_rank)
    return node_rank
