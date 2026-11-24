import math
import random
import time
from typing import Tuple, Any, List, Optional, Callable

import os
import sys
import warnings

import numpy as np
import torch
from einops import rearrange, reduce, repeat

# from flashinfer.utils import make_l2_flush, round_up

def set_seed(random_seed):
    """
    Set random seed for reproducibility during testing.

    Args:
        random_seed (int): Random seed to set.

    Returns:
        None
    """
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

    if torch.npu.is_available():
        torch.npu.manual_seed(random_seed)
        torch.npu.manual_seed_all(random_seed)

def sleep_after_kernel_run(execution_time):
    """
    Sleep after kernel run. Dynamically adjust sleep time up to 1 sec based on execution time.

    Args:
        execution_time (float): Kernel execution time in milliseconds.

    Returns:
        None
    """
    if not math.isinf(execution_time):
        sleep_time = np.min([execution_time / 200, 1.0])
    else:
        sleep_time = 0.01
    time.sleep(sleep_time)
    return

def aggregate_gpu_time_across_ranks(x, op):
    """
    Aggregate GPU time across ranks.

    Args:
        x (int | float | List[int] | List[float]): GPU time to aggregate.
        op (Callable): Operation to perform across ranks.

    Returns:
        int | float | List[int] | List[float]: Aggregated GPU time.
    """
    if torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        if world_size > 1:
            x_all = [None] * world_size
            torch.distributed.all_gather_object(x_all, x)
            if isinstance(x, list):
                x = [op(val) for val in zip(*x_all, strict=True)]
            else:
                x = op(x_all)
    return x

def bench_gpu_time_legacy(
    fn,
    dry_run_iters: int = None,
    repeat_iters: int = None,
    dry_run_time_ms: int = 25,
    repeat_time_ms: int = 100,
    l2_flush: Optional[bool] = None,  # Deprecated. Use cold_l2_cache instead
    l2_flush_size_mb: Optional[int] = None,  # Deprecated. Use cold_l2_cache instead
    l2_flush_device: Optional[str] = None,  # Deprecated. Use cold_l2_cache instead
    sleep_after_run: bool = False,
    input_args: Tuple = (),
    input_kwargs: Optional[dict] = None,
    cold_l2_cache: bool = True,
    aggregate_op: Callable = max,
):
    """
    Benchmark kernel execution time using CUDA events (no CUDA graphs).

    This is the simplest benchmarking method. Best suited for kernels where launch overhead
    is negligible compared to execution time.

    The function performs:
    1. A quick estimation phase (5 iterations) to determine iteration counts
    2. Dry-run warmup iterations (not measured)
    3. Measured iterations with per-iteration timing via CUDA events

    Iteration counts can be specified directly or derived from target durations:
    - If dry_run_iters/repeat_iters are provided, those counts are used directly.
    - Otherwise, counts are computed from dry_run_time_ms/repeat_time_ms.

    Args:
        fn (Callable): The kernel function to benchmark.
        dry_run_iters (int, optional): Number of warmup iterations (not timed).
            If None, computed from dry_run_time_ms.
        repeat_iters (int, optional): Number of measured iterations.
            If None, computed from repeat_time_ms.
        dry_run_time_ms (int): Target warmup duration in ms (default: 25).
        repeat_time_ms (int): Target measurement duration in ms (default: 100).
        sleep_after_run (bool): If True, sleep briefly after each iteration to
            reduce thermal throttling (default: False).
        input_args (tuple): Positional arguments to pass to fn.
        input_kwargs (dict, optional): Keyword arguments to pass to fn.
        cold_l2_cache (bool): If True, flush L2 cache before each iteration to
            ensure cold-cache performance measurements (default: True).
        aggregate_op (Callable): Aggregate operation to perform across ranks (default: max).

    Returns:
        List[float]: Per-iteration execution times in milliseconds.

    Example:
        Basic usage:

        >>> def my_kernel(a, b):
        ...     return torch.matmul(a, b.T)
        >>> q = torch.randn(1024, 128, device="cuda")
        >>> k = torch.randn(1024, 128, device="cuda")
        >>> times = bench_gpu_time_with_cuda_event(
        ...     fn=my_kernel,
        ...     input_args=(q, k),
        ... )
        >>> print(f"Median time: {np.median(times):.3f} ms")

    Note:
        This method does NOT use CUDA graphs, so each iteration incurs kernel
        launch overhead. For microbenchmarking where launch latency matters,
        consider using ``bench_gpu_time_with_cudagraph`` instead.

    .. deprecated:: 0.6.0
        The ``l2_flush``, ``l2_flush_size_mb``, and ``l2_flush_device`` parameters
        are deprecated. Use ``cold_l2_cache`` instead.
    """
    if input_kwargs is None:
        input_kwargs = {}

    ## TODO:
    ## Need support for "l2_cache".

    # Handle deprecated parameters
    # if any(p is not None for p in [l2_flush, l2_flush_size_mb, l2_flush_device]):
    #     warnings.warn(
    #         "l2_flush, l2_flush_size_mb, and l2_flush_device are deprecated. "
    #         "Use cold_l2_cache instead.",
    #         category=DeprecationWarning,
    #         stacklevel=2,
    #     )
    #     _do_l2_flush = l2_flush if l2_flush is not None else True
    #     _l2_flush_size_mb = l2_flush_size_mb if l2_flush_size_mb is not None else 256
    #     _l2_flush_device = l2_flush_device if l2_flush_device is not None else "cuda"
    # else:
    #     _do_l2_flush = cold_l2_cache
    #     # Dynamically determine L2 flush size and device
    #     _l2_flush_device = _infer_device_from_tensors(input_args, input_kwargs, "cuda")
    #     l2_size = get_l2_cache_size(_l2_flush_device)
    #     # Use 2x L2 size to ensure complete flush
    #     _l2_flush_size_mb = (l2_size * 2) // (1024 * 1024)

    # Check if args are provided (determines how we call fn)
    has_args = bool(input_args) or bool(input_kwargs)

    def call_fn():
        if has_args:
            fn(*input_args, **input_kwargs)
        else:
            fn()

    start_event = torch.npu.Event(enable_timing=True)
    end_event = torch.npu.Event(enable_timing=True)

    ## TODO:
    ## Need support for "l2_cache".

    # _l2_flush_fn = None
    # if _do_l2_flush:
    #     l2_flush_size = int(_l2_flush_size_mb) * 1024 * 1024
    #     _l2_flush_fn = make_l2_flush(l2_flush_size, _l2_flush_device)

    ## Estimate kernel execution time by running the kernel 5 times
    measurement_iters = 5
    torch.npu.synchronize()
    call_fn()  # Call once to exclude initial overhead
    torch.npu.synchronize()
    start_event.record()
    for _ in range(measurement_iters):
        # if _l2_flush_fn is not None:
        #     _l2_flush_fn()
        call_fn()
    end_event.record()
    torch.npu.synchronize()
    estimated_kernel_execution_time = aggregate_gpu_time_across_ranks(
        start_event.elapsed_time(end_event) / measurement_iters,
        aggregate_op,
    )

    ## Set dry run and repeat iterations
    if dry_run_iters is None:
        dry_run_iters = max(1, int(dry_run_time_ms / estimated_kernel_execution_time))
    if repeat_iters is None:
        repeat_iters = max(1, int(repeat_time_ms / estimated_kernel_execution_time))

    # Dry runs
    torch.npu.synchronize()
    for _ in range(dry_run_iters):
        # if _l2_flush_fn is not None:
        #     _l2_flush_fn()
        call_fn()
    torch.npu.synchronize()

    # Actual run
    start_events = [torch.npu.Event(enable_timing=True) for _ in range(repeat_iters)]
    end_events = [torch.npu.Event(enable_timing=True) for _ in range(repeat_iters)]
    torch.npu.synchronize()
    for iter_idx in range(repeat_iters):
        # if _l2_flush_fn is not None:
        #     _l2_flush_fn()
        start_events[iter_idx].record()
        call_fn()
        end_events[iter_idx].record()

        if sleep_after_run:
            sleep_after_kernel_run(estimated_kernel_execution_time)

    # Synchronize once outside of the loop to avoid synchronization overhead
    torch.npu.synchronize()
    measured_times = []
    for iter_idx in range(repeat_iters):
        measured_times.append(start_events[iter_idx].elapsed_time(end_events[iter_idx]))
    measured_times = aggregate_gpu_time_across_ranks(measured_times, aggregate_op)
    return measured_times

def bench_gpu_time(
    fn,
    dry_run_iters: int = None,
    repeat_iters: int = None,
    dry_run_time_ms: int = 25,
    repeat_time_ms: int = 100,
    l2_flush: Optional[bool] = None,  # Deprecated. Use cold_l2_cache instead
    l2_flush_size_mb: Optional[int] = None,  # Deprecated. Use cold_l2_cache instead
    l2_flush_device: Optional[str] = None,  # Deprecated. Use cold_l2_cache instead
    sleep_after_run: bool = False,
    enable_cupti: bool = False,
    use_cuda_graph: bool = False,
    num_iters_within_graph: int = 10,
    input_args: Tuple = (),
    input_kwargs: Optional[dict] = None,
    cold_l2_cache: bool = True,
    aggregate_op: Callable = max,
):
    """
    Unified GPU benchmarking interface with configurable timing backends.

    This is the recommended entry point for GPU kernel benchmarking. It provides
    a single interface that dispatches to the appropriate timing implementation
    based on the configuration flags.

    **Timing Backends** (in order of precedence):

    1. **CUPTI** (``enable_cupti=True``): Most accurate, measures pure GPU kernel
       time via hardware profiling. Requires cupti-python >= 13.
    2. **CUDA Graphs** (``use_cuda_graph=True``): Amortizes launch overhead by
       capturing and replaying multiple kernel calls. Good balance of accuracy
       and availability.
    3. **CUDA Events** (default): Simplest method, measures launch + execution.
       Available everywhere but includes CPU overhead.

    **Cold-L2 Strategy** (automatically selected based on timing backend):

    .. list-table::
       :header-rows: 1

       * - Timing Backend
         - Cold-L2 Strategy
         - How it Works
       * - CUPTI
         - L2 Flush
         - Flush L2 cache before each iter
       * - CUDA Events (no CUDA Graphs)
         - L2 Flush
         - Flush L2 cache before each iter
       * - CUDA Events + CUDA Graphs
         - Rotating Buffers
         - Clone GPU tensors in input_args/input_kwargs and rotate through them
        use_cuda_graph (bool): If True, use CUDA graph timing (default: False).
        num_iters_within_graph (int): Kernel calls per graph (CUDA graph mode only,
            default: 10).
        input_args (tuple): Positional arguments to pass to fn.
        input_kwargs (dict, optional): Keyword arguments to pass to fn.
        cold_l2_cache (bool): If True, ensure cold L2 cache for each iteration
            (default: True). The strategy is automatically selected based on timing
            backend.
        aggregate_op (Callable): Aggregate operation to perform across ranks (default: max).

    Returns:
        List[float]: Per-iteration execution times in milliseconds.

    Example:
        Simple benchmarking with CUDA events (default):

        >>> times = bench_gpu_time(fn=lambda: my_kernel())
        >>> print(f"Median: {np.median(times):.3f} ms")

    Example:
        CUDA graph benchmarking for reduced launch overhead:

        >>> def run_kernel(x, y, out):
        ...     my_memory_bound_kernel(x, y, out)
        >>> times = bench_gpu_time(
        ...     fn=run_kernel,
        ...     input_args=(x, y, out),
        ...     use_cuda_graph=True,
        ... )

    Example:
        CUPTI benchmarking for most accurate GPU kernel time:

        >>> times = bench_gpu_time(
        ...     fn=run_kernel,
        ...     input_args=(x, y, out),
        ...     enable_cupti=True,
        ... )

    See Also:
        - ``bench_gpu_time_with_cuda_event``: Direct CUDA event timing.
        - ``bench_gpu_time_with_cudagraph``: Direct CUDA graph timing.
        - ``bench_gpu_time_with_cupti``: Direct CUPTI timing.

    .. deprecated:: 0.6.0
        The ``l2_flush``, ``l2_flush_size_mb``, and ``l2_flush_device``
        parameters are deprecated. Use ``cold_l2_cache`` instead.
    """
    # Handle deprecated parameters

    ## TODO:
    ## Need support for "l2_cache".
    _cold_l2_cache = cold_l2_cache
    # if any(p is not None for p in [l2_flush, l2_flush_size_mb, l2_flush_device]):
    #     warnings.warn(
    #         "l2_flush, l2_flush_size_mb, and l2_flush_device are deprecated. "
    #         "Use cold_l2_cache instead.",
    #         category=DeprecationWarning,
    #         stacklevel=2,
    #     )
    #     # If l2_flush was explicitly set, use it as the cold_l2_cache value
    #     _cold_l2_cache = l2_flush if l2_flush is not None else cold_l2_cache
    # else:
    #     _cold_l2_cache = cold_l2_cache

    # if enable_cupti:
    #     return bench_gpu_time_with_cupti(
    #         fn=fn,
    #         dry_run_iters=dry_run_iters,
    #         repeat_iters=repeat_iters,
    #         dry_run_time_ms=dry_run_time_ms,
    #         repeat_time_ms=repeat_time_ms,
    #         sleep_after_run=sleep_after_run,
    #         use_cuda_graph=use_cuda_graph,
    #         input_args=input_args,
    #         input_kwargs=input_kwargs,
    #         cold_l2_cache=_cold_l2_cache,
    #         aggregate_op=aggregate_op,
    #     )
    # if use_cuda_graph:
    #     return bench_gpu_time_with_cudagraph(
    #         fn=fn,
    #         dry_run_iters=dry_run_iters,
    #         repeat_iters=repeat_iters,
    #         dry_run_time_ms=dry_run_time_ms,
    #         repeat_time_ms=repeat_time_ms,
    #         num_iters_within_graph=num_iters_within_graph,
    #         sleep_after_run=sleep_after_run,
    #         input_args=input_args,
    #         input_kwargs=input_kwargs,
    #         cold_l2_cache=_cold_l2_cache,
    #         aggregate_op=aggregate_op,
    #     )
    return bench_gpu_time_legacy(
        fn=fn,
        dry_run_iters=dry_run_iters,
        repeat_iters=repeat_iters,
        dry_run_time_ms=dry_run_time_ms,
        repeat_time_ms=repeat_time_ms,
        sleep_after_run=sleep_after_run,
        input_args=input_args,
        input_kwargs=input_kwargs,
        cold_l2_cache=_cold_l2_cache,
        aggregate_op=aggregate_op,
    )