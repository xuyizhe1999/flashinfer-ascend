from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

import flashinfer
from flashinfer.testing.utils import (
    bench_gpu_time,
    # dequantize_fp8,
    # quantize_fp8,
)

from .flashinfer_benchmark_utils import (
    # check_batch_invariance,
    dtype_str_to_torch_dtype,
    get_device,
    print_perf_metrics,
    warn_if_pdl_unsupported,
)

def run_gemm_test(args):
    """
    Run a gemm test.

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    if args.routine == "mm_bf16":
        return testMmBf16(args)
    # elif args.routine == "mm_fp4":
    #     return testMmFp4(args)
    # elif args.routine == "mm_bf16_fp4":
    #     return testMmBf16Fp4(args)
    # elif args.routine == "mm_mxfp8":
    #     return testMmMxfp8(args)
    elif args.routine == "bmm_fp8":
        return testBmmFp8(args)
    else:
        raise ValueError(f"Unsupported routine: {args.routine}")

def testMmBf16(args):
    """
    Test mm_bf16 API.

    This test:
    1. Generates random BF16 input tensors
    2. Runs mm_bf16 with specified backend
    3. Runs reference check (torch.mm)
    4. Measures performance metrics (TFLOPS, TB/sec)

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    if args.verbose >= 1:
        print("[INFO] Running testMmBf16")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    ## Parse input arguments
    backends = args.backends
    m = args.m
    n = args.n
    k = args.k
    use_bias = getattr(args, "bias", False)
    use_pdl = getattr(args, "enable_pdl", False)
    is_cuda_graph_compatible = not args.no_cuda_graph
    run_refcheck = args.refcheck

    ## TODO:
    ## Need future support for "autotune".

    # autotune_supported_backends = [
    #     "cudnn",
    #     "cutlass",
    #     "tgv",
    #     "cublaslt",
    #     "tinygemm",
    #     "cute-dsl",
    #     "auto",
    #     "batch_invariant",
    # ]

    res = []

    out_dtype = dtype_str_to_torch_dtype(args.out_dtype)
    if out_dtype not in [torch.bfloat16, torch.float16, torch.float32]:
        raise ValueError(
            f"Unsupported output dtype: {args.out_dtype}. Supported dtypes are bfloat16, float16, and float32."
        )

    ## Prepare input tensors
    # a: (m, k), row-major
    a = torch.randn([m, k], device=device, dtype=torch.bfloat16)
    # b: (n, k) then transpose to (k, n) for column-major layout
    b = torch.randn([n, k], device=device, dtype=torch.bfloat16).transpose(-2, -1)

    bias = None
    if use_bias:
        bias = torch.randn([n], device=device, dtype=torch.bfloat16)

    if args.verbose >= 2:
        print(f"[VVERBOSE] {a.shape = }")
        print(f"[VVERBOSE] {a.dtype = }")
        print(f"[VVERBOSE] {b.shape = }")
        print(f"[VVERBOSE] {b.dtype = }")
        if bias is not None:
            print(f"[VVERBOSE] {bias.shape = }")
            print(f"[VVERBOSE] {bias.dtype = }")
        print(f"[VVERBOSE] {use_pdl = }")

    def run_backend(backend, a, b, bias, use_pdl, out_dtype):
        if backend in [
            # nvidia backends should be parsed to npu backends automatically
            "cudnn",

            "cann",
            "ascendc",
            "triton",
            "auto",
        ]:
            return flashinfer.mm_bf16(
                a=a,
                b=b,
                bias=bias,
                pdl=use_pdl,
                out_dtype=out_dtype,
                backend=backend,
            )
        ## TODO:
        ## Need "batch_invariant" feature support.

        # elif backend == "batch_invariant":
        #     return flashinfer.mm_bf16(
        #         a=a,
        #         b=b,
        #         bias=bias,
        #         pdl=use_pdl,
        #         out_dtype=out_dtype,
        #         backend="auto",
        #         batch_invariant=True,
        #     )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    # Programmatically filter backends
    backends_to_remove = []
    for backend in backends:
        # Skip autotune check for now (handled separately below)

        ## TODO:
        ## Need future support for "autotune".

        # if (
        #     getattr(args, "autotune", False)
        #     and backend not in autotune_supported_backends
        # ):
        #     print(f"[INFO] {backend} backend does not support autotune")
        #     backends_to_remove.append(backend)
        #     continue

        try:
            run_backend(backend, a, b, bias, use_pdl, out_dtype)
        except Exception as e:
            print(
                f"[INFO] {backend} backend does not support this configuration: {type(e).__name__}: {e}"
            )
            backends_to_remove.append(backend)

    # Remove unsupported backends
    for backend in backends_to_remove:
        backends.remove(backend)

    if len(backends) == 0:
        print("[ERROR] No backends passed validation. Exiting.")
        return res

    has_reference_output = False
    reference_output_base = None
    if run_refcheck:
        reference_output_base = torch.mm(a, b).to(out_dtype)
        has_reference_output = True

    ## TODO:
    ## Need future support for "autotune".

    # cache_path = getattr(args, "autotune_cache", None)
    # if getattr(args, "autotune", False):
    #     warmup_iters = (
    #         args.dry_run_iters if args.dry_run_iters and args.dry_run_iters > 0 else 10
    #     )
    #     for cur_backend in backends:
    #         if cur_backend in autotune_supported_backends:
    #             if args.verbose >= 1:
    #                 print(f"[INFO] Autotune warmup for mm_bf16: {warmup_iters} iters")
    #             with autotune(True, cache=cache_path):
    #                 for _ in range(warmup_iters):
    #                     run_backend(cur_backend, a, b, bias, use_pdl, out_dtype)
    # elif cache_path:
    #     with autotune(False, cache=cache_path):
    #         pass

    # Storage for timing results and outputs
    backend_times = {backend: [] for backend in backends}
    outputs = {}
    for cur_backend in backends:
        if run_refcheck:
            outputs[cur_backend] = run_backend(
                cur_backend, a, b, bias, use_pdl, out_dtype
            ).detach()
        backend_times[cur_backend] = bench_gpu_time(
            fn=run_backend,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            sleep_after_run=True,
            enable_cupti=args.use_cupti,
            use_cuda_graph=is_cuda_graph_compatible,
            cold_l2_cache=True,
            input_args=(cur_backend, a, b, bias, use_pdl, out_dtype),
        )

    ## TODO:
    ## Need "batch_invariant" feature support.

    # if run_refcheck and "batch_invariant" in outputs:
    #     # Bitwise: the same rows of a alone / in a sub-batch / permuted give the same bits.
    #     def run_rows(rows):
    #         return run_backend(
    #             "batch_invariant", a.index_select(0, rows), b, bias, use_pdl, out_dtype
    #         )

    #     check_batch_invariance(
    #         "mm_bf16/batch_invariant",
    #         run_rows,
    #         m,
    #         outputs["batch_invariant"],
    #         verbose=args.verbose,
    #     )

    tested_backends = list(outputs.keys())
    tested_outputs = list(outputs.values())
    if len(tested_backends) > 0:
        if run_refcheck and has_reference_output:
            for i in range(len(tested_backends)):
                ## TODO:
                ## Need support for "bias" feature.
                reference_output = reference_output_base
                # if (
                #     tested_backends[i]
                #     in (
                #         "cudnn",
                #         "tgv",
                #         "tinygemm",
                #         "cute-dsl",
                #         "batch_invariant",
                #     )
                #     and bias is not None
                # ):
                #     reference_output = reference_output_base + bias.unsqueeze(0).to(
                #         out_dtype
                #     )
                # else:
                #     reference_output = reference_output_base

                cos_sim = F.cosine_similarity(
                    reference_output.reshape(-1),
                    tested_outputs[i].reshape(-1),
                    dim=0,
                )
                if cos_sim < 0.99:
                    print(
                        f"[ERROR] Output tensor mismatch from backend {tested_backends[i]} with cos_sim={cos_sim}"
                    )
                    if not args.allow_output_mismatch:
                        raise AssertionError(
                            f"[ERROR] Backend {tested_backends[i]} output mismatch with cos_sim={cos_sim}"
                        )

    for backend in backends:
        ## TODO:
        ## Need future support for "autotune".
        backend_name = backend
        # backend_name = backend + (
        #     "_autotune"
        #     if (
        #         getattr(args, "autotune", False)
        #         and backend in autotune_supported_backends
        #     )
        #     else ""
        # )
        if len(backend_times[backend]) > 0:
            median_time = np.median(backend_times[backend])
            std_time = np.std(backend_times[backend])
            problem_flops = 2 * m * n * k
            if use_bias:
                problem_flops += m * n  # bias addition
            problem_bytes = (
                m * k * torch.bfloat16.itemsize
                + k * n * torch.bfloat16.itemsize
                + m * n * out_dtype.itemsize
            )
            if use_bias:
                problem_bytes += n * torch.bfloat16.itemsize
            tflops = problem_flops / (10**9 * median_time)  # in TFLOPs/sec
            tb_per_sec = problem_bytes / (10**9 * median_time)  # in TB/sec
            print_perf_metrics(backend_name, median_time, std_time, tflops, tb_per_sec)

            if args.output_path is not None:
                cur_res = defaultdict(str)
                cur_res["routine"] = args.routine
                cur_res["median_time"] = median_time
                cur_res["std_time"] = std_time
                cur_res["tflops"] = tflops
                cur_res["tb_per_sec"] = tb_per_sec
                cur_res["m"] = m
                cur_res["n"] = n
                cur_res["k"] = k
                cur_res["out_dtype"] = str(out_dtype)
                cur_res["backend"] = backend_name
                cur_res["bias"] = use_bias
                cur_res["enable_pdl"] = use_pdl
                cur_res["case_tag"] = args.case_tag
                res.append(cur_res)
    return res

def testBmmFp8(args):
    """
    Test bmm_fp8 API.

    This test:
    1. Generates random input tensors
    2. Quantizes input tensors to FP8
    3. Runs bmm_fp8
    4. Runs reference check
    5. Measures performance metrics (TFLOPS, TB/sec)

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        dict: List of dictionaries containing performance results
    """
    warn_if_pdl_unsupported(args, args.routine)
    if args.verbose >= 1:
        print("[INFO] Running testBmmFp8")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    ## Parse input arguments
    backends = args.backends
    batch_size = args.batch_size
    m = args.m
    n = args.n
    k = args.k
    input_dtype = args.input_dtype
    mat2_dtype = args.mat2_dtype
    res_dtype = args.out_dtype
    backends = args.backends
    is_cuda_graph_compatible = not args.no_cuda_graph
    run_refcheck = args.refcheck

    ## TODO:
    ## Need future support for "autotune".

    # autotune_supported_backends = [
    #     "cudnn",
    #     "cutlass",
    #     "tgv",
    #     "cublaslt",
    #     "tinygemm",
    #     "cute-dsl",
    #     "auto",
    #     "batch_invariant",
    # ]

    res = []

    input_dtype = dtype_str_to_torch_dtype(args.input_dtype)
    if input_dtype not in [torch.float8_e4m3fn, torch.float8_e5m2]:
        raise ValueError(
            f"Unsupported input dtype: {input_dtype}. Supported dtypes are fp8_e4m3 and fp8_e5m2."
        )

    mat2_dtype = dtype_str_to_torch_dtype(args.mat2_dtype)
    if mat2_dtype not in [torch.float8_e4m3fn, torch.float8_e5m2]:
        raise ValueError(
            f"Unsupported mat2 dtype: {mat2_dtype}. Supported dtypes are fp8_e4m3 and fp8_e5m2."
        )

    res_dtype = dtype_str_to_torch_dtype(args.out_dtype)
    if res_dtype not in [torch.bfloat16, torch.float16]:
        raise ValueError(
            f"Unsupported res dtype: {res_dtype}. Supported dtypes are bfloat16 and float16."
        )
    ## Done parsing input arguments

    ## Prepare input tensors
    input = torch.randn([batch_size, m, k], device=device, dtype=torch.bfloat16)
    input_fp8, input_inv_s = to_float8(input, dtype=input_dtype)

    mat2 = torch.randn(
        [batch_size, n, k], device=device, dtype=torch.bfloat16
    ).transpose(-2, -1)
    mat2_fp8, mat2_inv_s = to_float8(mat2, dtype=mat2_dtype)

    if args.verbose >= 2:
        print(f"[VVERBOSE] {input_fp8.shape = }")
        print(f"[VVERBOSE] {input_fp8.dtype = }")
        print(f"[VVERBOSE] {mat2_fp8.shape = }")
        print(f"[VVERBOSE] {mat2_fp8.dtype = }")
        print(f"[VVERBOSE] {input_inv_s = }")
        print(f"[VVERBOSE] {input_inv_s.dtype = }")
        print(f"[VVERBOSE] {mat2_inv_s = }")
        print(f"[VVERBOSE] {mat2_inv_s.dtype = }")

    def run_backend(backend, input_fp8, mat2_fp8, input_inv_s, mat2_inv_s):
        ## TODO:
        ## Need "batch_invariant" feature support.

        # if backend == "batch_invariant":
        #     return flashinfer.gemm.bmm_fp8(
        #         A=input_fp8,
        #         B=mat2_fp8,
        #         A_scale=input_inv_s,
        #         B_scale=mat2_inv_s,
        #         dtype=res_dtype,
        #         backend="auto",
        #         batch_invariant=True,
        #     )
        return flashinfer.gemm.bmm_fp8(
            A=input_fp8,
            B=mat2_fp8,
            A_scale=input_inv_s,
            B_scale=mat2_inv_s,
            dtype=res_dtype,
            backend=backend,
        )

    # Programmatically filter backends: rely on bmm_fp8's @backend_requirement
    # support checks (instead of a hard-coded compute-capability table) by
    # probing each backend with a trial call.
    backends_to_remove = []
    for backend in backends:
        ## TODO:
        ## Need future support for "autotune".

        # if (
        #     getattr(args, "autotune", False)
        #     and backend not in autotune_supported_backends
        # ):
        #     print(f"[INFO] {backend} backend does not support autotune")
        #     backends_to_remove.append(backend)
        #     continue

        try:
            run_backend(backend, input_fp8, mat2_fp8, input_inv_s, mat2_inv_s)
        except Exception as e:
            print(
                f"[INFO] {backend} backend does not support this configuration: {type(e).__name__}: {e}"
            )
            backends_to_remove.append(backend)

    for backend in backends_to_remove:
        backends.remove(backend)

    if len(backends) == 0:
        print("[ERROR] No backends passed validation. Exiting.")
        return res

    has_reference_output = False
    if run_refcheck:
        reference_output = torch.bmm(input, mat2)
        has_reference_output = True

    ## TODO:
    ## Need future support for "autotune".

    # cache_path = getattr(args, "autotune_cache", None)
    # if getattr(args, "autotune", False):
    #     warmup_iters = (
    #         args.dry_run_iters if args.dry_run_iters and args.dry_run_iters > 0 else 10
    #     )
    #     for cur_backend in backends:
    #         if cur_backend in autotune_supported_backends:
    #             if args.verbose >= 1:
    #                 print(f"[INFO] Autotune warmup for bmm_fp8: {warmup_iters} iters")
    #             with autotune(True, cache=cache_path):
    #                 for _ in range(warmup_iters):
    #                     run_backend(
    #                         cur_backend, input_fp8, mat2_fp8, input_inv_s, mat2_inv_s
    #                     )
    # elif cache_path:
    #     with autotune(False, cache=cache_path):
    #         pass

    # Storage for timing results and outputs
    backend_times = {backend: [] for backend in backends}
    outputs = {}
    for cur_backend in backends:
        if run_refcheck:
            outputs[cur_backend] = run_backend(
                cur_backend, input_fp8, mat2_fp8, input_inv_s, mat2_inv_s
            ).detach()
        backend_times[cur_backend] = bench_gpu_time(
            fn=run_backend,
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            sleep_after_run=True,
            enable_cupti=args.use_cupti,
            use_cuda_graph=is_cuda_graph_compatible,
            cold_l2_cache=True,
            input_args=(cur_backend, input_fp8, mat2_fp8, input_inv_s, mat2_inv_s),
        )

    tested_backends = list(outputs.keys())
    tested_outputs = list(outputs.values())
    if len(tested_backends) > 0:
        if run_refcheck and has_reference_output:
            if reference_output.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                print(
                    "[INFO] Reference output is FP8. Converting to float32 for reference check."
                )
                reference_output = reference_output.to(torch.float32)
                tested_outputs = [output.to(torch.float32) for output in tested_outputs]
            for i in range(len(tested_backends)):
                cos_sim = F.cosine_similarity(
                    reference_output.reshape(-1),
                    tested_outputs[i].reshape(-1),
                    dim=0,
                )
                if cos_sim < 0.99:
                    print(
                        f"[ERROR] Output tensor mismatch between backends {tested_backends[0]} and {tested_backends[i]}"
                    )
                    if not args.allow_output_mismatch:
                        raise AssertionError(
                            f"[ERROR] Backend {tested_backends[i]} output mismatch with cos_sim={cos_sim}"
                        )

    ## TODO:
    ## Need "batch_invariant" feature support.

    # if run_refcheck and "batch_invariant" in outputs:
    #     # Bitwise: the same rows of every batch element (the M axis) alone / in a sub-batch / permuted.
    #     def run_rows(rows):
    #         return run_backend(
    #             "batch_invariant",
    #             input_fp8.index_select(1, rows),
    #             mat2_fp8,
    #             input_inv_s,
    #             mat2_inv_s,
    #         )

    #     check_batch_invariance(
    #         "bmm_fp8/batch_invariant",
    #         run_rows,
    #         m,
    #         outputs["batch_invariant"],
    #         row_dim=1,
    #         verbose=args.verbose,
    #     )

    for backend in backends:
        ## TODO:
        ## Need future support for "autotune".
        backend_name = backend
        # backend_name = backend + (
        #     "_autotune"
        #     if (
        #         getattr(args, "autotune", False)
        #         and backend in autotune_supported_backends
        #     )
        #     else ""
        # )
        if len(backend_times[backend]) > 0:
            median_time = np.median(backend_times[backend])
            std_time = np.std(backend_times[backend])
            problem_flops = 2 * m * n * k * batch_size
            problem_bytes = (
                m * k * input_dtype.itemsize
                + n * k * mat2_dtype.itemsize
                + m * n * res_dtype.itemsize
            )
            tflops = problem_flops / (10**9 * median_time)  # in TFLOPs/sec
            tb_per_sec = problem_bytes / (10**9 * median_time)  # in TB/sec
            print_perf_metrics(backend_name, median_time, std_time, tflops, tb_per_sec)

            if args.output_path is not None:
                cur_res = defaultdict(str)
                cur_res["batch_size"] = batch_size
                cur_res["routine"] = args.routine
                cur_res["median_time"] = median_time
                cur_res["std_time"] = std_time
                cur_res["tflops"] = tflops
                cur_res["tb_per_sec"] = tb_per_sec
                cur_res["m"] = m
                cur_res["n"] = n
                cur_res["k"] = k
                cur_res["input_dtype"] = input_dtype
                cur_res["mat2_dtype"] = mat2_dtype
                cur_res["out_dtype"] = res_dtype
                cur_res["backend"] = backend_name
                cur_res["case_tag"] = args.case_tag
                res.append(cur_res)
    return res

def to_float8(x, dtype=torch.float8_e4m3fn):
    finfo = torch.finfo(dtype)
    min_val, max_val = x.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)
    scale = finfo.max / amax
    x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
    return x_scl_sat.to(dtype), scale.float().reciprocal()

# def testMmFp4(args):
#     """
#     Test mm_fp4 API.

#     This test:
#     1. Generates random input tensors
#     2. Quantizes input tensors to FP4
#     3. Runs mm_fp4
#     4. Runs reference check
#     5. Measures performance metrics (TFLOPS, TB/sec)

#     Args:
#         args: Parsed command line arguments containing test configuration

#     Returns:
#         dict: List of dictionaries containing performance results
#     """
#     if args.verbose >= 1:
#         print("[INFO] Running testMmFp4")
#         print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

#     device = get_device(args)
#     if args.generate_repro_command:
#         print(
#             f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
#         )

#     ## Parse input arguments
#     backends = args.backends
#     m = args.m
#     n = args.n
#     k = args.k
#     res_dtype = args.out_dtype
#     backends = args.backends
#     is_cuda_graph_compatible = not args.no_cuda_graph
#     run_refcheck = args.refcheck
#     use_128x4_sf_layout = args.use_128x4_sf_layout
#     use_nvfp4 = args.use_nvfp4

#     ## TODO:
#     ## Need future support for "autotune".

#     # autotune_supported_backends = [
#     #     "cudnn",
#     #     "cutlass",
#     #     "tgv",
#     #     "cublaslt",
#     #     "tinygemm",
#     #     "cute-dsl",
#     #     "auto",
#     #     "batch_invariant",
#     # ]

#     res = []

#     res_dtype = dtype_str_to_torch_dtype(args.out_dtype)
#     if res_dtype not in [torch.bfloat16, torch.float16]:
#         raise ValueError(
#             f"Unsupported res dtype: {res_dtype}. Supported dtypes are bfloat16 and float16."
#         )

#     input = torch.randn([m, k], device=device, dtype=torch.bfloat16)
#     mat2 = torch.randn([n, k], device=device, dtype=torch.bfloat16)

#     ## TODO:
#     ## Need future support for "nvfp4".

#     # a_sf_layout = (
#     #     flashinfer.SfLayout.layout_128x4
#     #     if use_128x4_sf_layout
#     #     else flashinfer.SfLayout.layout_8x4
#     # )

#     global_sf_input = (448 * 6) / input.float().abs().nan_to_num().max()
#     global_sf_mat2 = (448 * 6) / mat2.float().abs().nan_to_num().max()

#     ## TODO:
#     ## Need future support for "nvfp4".

#     input_fp4, input_inv_s = flashinfer.mxfp4_quantize(input)
#     mat2_fp4, mat2_inv_s = flashinfer.mxfp4_quantize(mat2)
#     # if use_nvfp4:
#     #     input_fp4, input_inv_s = flashinfer.nvfp4_quantize(
#     #         input, global_sf_input, sfLayout=a_sf_layout, do_shuffle=False
#     #     )
#     #     mat2_fp4, mat2_inv_s = flashinfer.nvfp4_quantize(
#     #         mat2,
#     #         global_sf_mat2,
#     #         sfLayout=flashinfer.SfLayout.layout_128x4,
#     #         do_shuffle=False,
#     #     )
#     # else:  # mxfp4
#     #     input_fp4, input_inv_s = flashinfer.mxfp4_quantize(input)
#     #     mat2_fp4, mat2_inv_s = flashinfer.mxfp4_quantize(mat2)

#     # The trtllm backend consumes a shuffled weight/scale layout. Prepare it
#     # only when that backend is requested: the shuffle requires n to be a
#     # multiple of 128, and an unaligned n must not fail the other backends.
#     mat2_fp4_trtllm, mat2_inv_s_trtllm = None, None
#     if "trtllm" in backends:
#         try:
#             mat2_fp4_trtllm, mat2_inv_s_trtllm = flashinfer.nvfp4_quantize(
#                 mat2,
#                 global_sf_mat2,
#                 sfLayout=flashinfer.SfLayout.layout_128x4,
#                 do_shuffle=True,
#             )
#         except Exception as e:
#             print(
#                 f"[INFO] trtllm backend does not support this configuration: {type(e).__name__}: {e}"
#             )
#             backends.remove("trtllm")

#     if args.verbose >= 2:
#         print(f"[VVERBOSE] {input_fp4.shape = }")
#         print(f"[VVERBOSE] {input_fp4.dtype = }")
#         print(f"[VVERBOSE] {mat2_fp4.shape = }")
#         print(f"[VVERBOSE] {mat2_fp4.dtype = }")

#     alpha = 1.0 / (global_sf_input * global_sf_mat2) if use_nvfp4 else None
#     # TODO: for MXFP4, we don't need a global scale, we should change the compile interface to make
#     # alpha optional.
#     alpha_for_cute_dsl_mxfp4 = (
#         torch.tensor([1.0], dtype=torch.float32, device=device)
#         if not use_nvfp4
#         else None
#     )
#     # Completed preparing inputs. Now programmatically filter backends
#     block_size = (
#         16 if use_nvfp4 else 32
#     )  # nvfp4 only supports 16; mxfp4 only supports 32.

#     def run_backend(
#         backend,
#         input_fp4,
#         mat2_fp4,
#         mat2_fp4_trtllm,
#         input_inv_s,
#         mat2_inv_s,
#         mat2_inv_s_trtllm,
#     ):
#         if backend == "batch_invariant":
#             return flashinfer.gemm.mm_fp4(
#                 a=input_fp4,
#                 b=mat2_fp4.T,
#                 a_descale=input_inv_s,
#                 b_descale=mat2_inv_s.T,
#                 alpha=alpha if use_nvfp4 else alpha_for_cute_dsl_mxfp4,
#                 out_dtype=res_dtype,
#                 block_size=block_size,
#                 use_8x4_sf_layout=not use_128x4_sf_layout,
#                 backend="auto",
#                 use_nvfp4=use_nvfp4,
#                 enable_pdl=args.enable_pdl,
#                 batch_invariant=True,
#             )
#         return flashinfer.gemm.mm_fp4(
#             a=input_fp4,
#             b=mat2_fp4.T if backend != "trtllm" else mat2_fp4_trtllm.T,
#             a_descale=input_inv_s,
#             b_descale=mat2_inv_s.T if backend != "trtllm" else mat2_inv_s_trtllm.T,
#             alpha=(
#                 alpha_for_cute_dsl_mxfp4 if backend in ("cute-dsl", "b12x") else alpha
#             ),
#             out_dtype=res_dtype,
#             block_size=block_size,
#             use_8x4_sf_layout=not use_128x4_sf_layout,
#             backend=backend,
#             use_nvfp4=use_nvfp4,
#             enable_pdl=args.enable_pdl,
#         )

#     backends_to_remove = []

#     for backend in backends:
#         # Skip autotune check for now (handled separately below)
#         if (
#             getattr(args, "autotune", False)
#             and backend not in autotune_supported_backends
#         ):
#             print(f"[INFO] {backend} backend does not support autotune")
#             backends_to_remove.append(backend)
#             continue

#         try:
#             run_backend(
#                 backend,
#                 input_fp4,
#                 mat2_fp4,
#                 mat2_fp4_trtllm,
#                 input_inv_s,
#                 mat2_inv_s,
#                 mat2_inv_s_trtllm,
#             )
#         except Exception as e:
#             print(
#                 f"[INFO] {backend} backend does not support this configuration: {type(e).__name__}: {e}"
#             )
#             backends_to_remove.append(backend)

#     # Remove unsupported backends
#     for backend in backends_to_remove:
#         backends.remove(backend)

#     if len(backends) == 0:
#         print("[ERROR] No backends passed validation. Exiting.")
#         return res

#     has_reference_output = False
#     if run_refcheck:
#         reference_output = torch.mm(input, mat2.T)
#         has_reference_output = True

#     cache_path = getattr(args, "autotune_cache", None)
#     if getattr(args, "autotune", False):
#         warmup_iters = (
#             args.dry_run_iters if args.dry_run_iters and args.dry_run_iters > 0 else 10
#         )
#         for cur_backend in backends:
#             if args.verbose >= 1:
#                 print(f"[INFO] Autotune warmup for mm_fp4: {warmup_iters} iters")
#             with autotune(True, cache=cache_path):
#                 for _ in range(warmup_iters):
#                     run_backend(
#                         cur_backend,
#                         input_fp4,
#                         mat2_fp4,
#                         mat2_fp4_trtllm,
#                         input_inv_s,
#                         mat2_inv_s,
#                         mat2_inv_s_trtllm,
#                     )
#     elif cache_path:
#         with autotune(False, cache=cache_path):
#             pass

#     # Storage for timing results and outputs
#     backend_times = {backend: [] for backend in backends}
#     outputs = {}
#     for cur_backend in backends:
#         if run_refcheck:
#             outputs[cur_backend] = run_backend(
#                 cur_backend,
#                 input_fp4,
#                 mat2_fp4,
#                 mat2_fp4_trtllm,
#                 input_inv_s,
#                 mat2_inv_s,
#                 mat2_inv_s_trtllm,
#             ).detach()
#         backend_times[cur_backend] = bench_gpu_time(
#             fn=run_backend,
#             dry_run_iters=args.dry_run_iters,
#             repeat_iters=args.num_iters,
#             sleep_after_run=True,
#             enable_cupti=args.use_cupti,
#             use_cuda_graph=is_cuda_graph_compatible,
#             cold_l2_cache=True,
#             input_args=(
#                 cur_backend,
#                 input_fp4,
#                 mat2_fp4,
#                 mat2_fp4_trtllm,
#                 input_inv_s,
#                 mat2_inv_s,
#                 mat2_inv_s_trtllm,
#             ),
#         )

#     tested_backends = list(outputs.keys())
#     tested_outputs = list(outputs.values())
#     if len(tested_backends) > 0:
#         if run_refcheck and has_reference_output:
#             for i in range(len(tested_backends)):
#                 cos_sim = F.cosine_similarity(
#                     reference_output.reshape(-1),
#                     tested_outputs[i].reshape(-1),
#                     dim=0,
#                 )
#                 if cos_sim < 0.97:
#                     print(
#                         f"[ERROR] Output tensor mismatch between backends {tested_backends[0]} and {tested_backends[i]}"
#                     )
#                     if not args.allow_output_mismatch:
#                         raise AssertionError(
#                             f"[ERROR] Backend {tested_backends[i]} output mismatch with cos_sim={cos_sim}"
#                         )

#     if run_refcheck and "batch_invariant" in outputs:

#         def run_rows(rows):
#             input_rows = input.index_select(0, rows)
#             if use_nvfp4:
#                 a_rows, a_rows_inv_s = flashinfer.nvfp4_quantize(
#                     input_rows, global_sf_input, sfLayout=a_sf_layout, do_shuffle=False
#                 )
#             else:
#                 a_rows, a_rows_inv_s = flashinfer.mxfp4_quantize(input_rows)
#             return run_backend(
#                 "batch_invariant",
#                 a_rows,
#                 mat2_fp4,
#                 mat2_fp4_trtllm,
#                 a_rows_inv_s,
#                 mat2_inv_s,
#                 mat2_inv_s_trtllm,
#             )

#         check_batch_invariance(
#             "mm_fp4/batch_invariant",
#             run_rows,
#             m,
#             outputs["batch_invariant"],
#             verbose=args.verbose,
#         )

#     for backend in backends:
#         backend_name = backend + (
#             "_autotune"
#             if (
#                 getattr(args, "autotune", False)
#                 and backend in autotune_supported_backends
#             )
#             else ""
#         )
#         if len(backend_times[backend]) > 0:
#             median_time = np.median(backend_times[backend])
#             std_time = np.std(backend_times[backend])
#             problem_flops = 2 * m * n * k
#             problem_bytes = (
#                 m * k * 0.5 + n * k * 0.5 + m * n * res_dtype.itemsize
#             )  # 0.5 for fp4
#             tflops = problem_flops / (10**9 * median_time)  # in TFLOPs/sec
#             tb_per_sec = problem_bytes / (10**9 * median_time)  # in TB/sec
#             print_perf_metrics(backend_name, median_time, std_time, tflops, tb_per_sec)

#             if args.output_path is not None:
#                 cur_res = defaultdict(str)
#                 cur_res["routine"] = args.routine
#                 cur_res["median_time"] = median_time
#                 cur_res["std_time"] = std_time
#                 cur_res["tflops"] = tflops
#                 cur_res["tb_per_sec"] = tb_per_sec
#                 cur_res["m"] = m
#                 cur_res["n"] = n
#                 cur_res["k"] = k
#                 cur_res["out_dtype"] = res_dtype
#                 cur_res["use_128x4_sf_layout"] = use_128x4_sf_layout
#                 cur_res["backend"] = backend_name
#                 cur_res["use_nvfp4"] = use_nvfp4
#                 cur_res["case_tag"] = args.case_tag
#                 res.append(cur_res)
#     return res

def parse_gemm_args(line, parser):
    """
    Parse command line arguments for gemm test configuration.

    Args:
        line: Command line arguments
        parser: ArgumentParser object already populated with shared arguments

    Returns:
        Parsed argument namespace
    """
    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=1,
        help="Batch size of test case.",
    )
    parser.add_argument(
        "--m", type=int, required=True, help="Number of rows in the first matrix."
    )
    parser.add_argument(
        "--n", type=int, required=True, help="Number of columns in the second matrix."
    )
    parser.add_argument(
        "--k",
        type=int,
        required=True,
        help="Number of columns in the first matrix and number of rows in the second matrix.",
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        required=False,
        default=128,
        help="Tile size for the gemm operation.",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        required=False,
        default=1,
        help="Group size for the group gemm operation.",
    )
    parser.add_argument(
        "--scale_major_mode",
        type=str,
        required=False,
        default="MN",
        choices=["MN", "K"],
        help="Scale major mode.",
    )
    parser.add_argument(
        "--input_dtype",
        type=str,
        required=False,
        default="fp8_e4m3",
        help="Data type of the input.",
    )
    parser.add_argument(
        "--mat2_dtype",
        type=str,
        required=False,
        default="fp8_e4m3",
        help="Data type of the mat2.",
    )
    parser.add_argument(
        "--out_dtype",
        type=str,
        required=False,
        default="bfloat16",
        help="Data type of the output.",
    )
    parser.add_argument(
        "--mma_sm",
        type=int,
        required=False,
        default=1,
        choices=[1, 2],
        help="How many SMs to use for the MMA operation, must be 1 or 2",
    )
    parser.add_argument(
        "--backends",
        type=str,
        required=False,
        nargs="+",
        default=["auto"],
        choices=[
            # nvidia backends should be parsed to npu backends automatically
            "cudnn",

            "cann",
            "ascendc",
            "triton",
            "auto",
        ],
        help=(
            "Kernel backends to test. Default: cudnn. `batch_invariant` runs the operator with "
            "batch_invariant=True (backend='auto'). With --refcheck its output is also checked bit for bit "
            "against the same rows computed alone, in a sub-batch and under a permutation."
        ),
    )
    parser.add_argument(
        "--use_128x4_sf_layout",
        action="store_true",
        help="Use 128x4 SF layout for the input and mat2.",
    )
    parser.add_argument(
        "--use_nvfp4",
        action="store_true",
        help="In mm_fp4, whether to use nvfp4 quantization or mxfp4 quantization, defaults to False.",
    )
    parser.add_argument(
        "--autotune",
        action="store_true",
        default=False,
        help=(
            "Enable autotuner warmup for supported routines (mm_fp4, bmm_fp8, mm_fp8, bmm_mxfp8, mm_mxfp8, mm_bf16, bmm_bf16; "
            "the batch_invariant backend of those and of gemm_fp8_nt_groupwise, which profiles the class's top configurations)."
        ),
    )
    parser.add_argument(
        "--bias",
        action="store_true",
        default=False,
        help="Use bias (enabled for mm_bf16 with TGV and TinyGEMM backends)",
    )

    args = parser.parse_args(line)
    has_backends_arg = any(
        token == "--backends" or token.startswith("--backends=") for token in line
    )
    has_input_dtype_arg = any(
        token == "--input_dtype" or token.startswith("--input_dtype=") for token in line
    )
    has_mat2_dtype_arg = any(
        token == "--mat2_dtype" or token.startswith("--mat2_dtype=") for token in line
    )
    if args.routine == "tinygemm_bf16":
        if not has_backends_arg:
            args.backends = ["tinygemm"]
        if not has_input_dtype_arg:
            args.input_dtype = "bfloat16"
        if not has_mat2_dtype_arg:
            args.mat2_dtype = "bfloat16"
    if args.routine == "mm_bf16_fp4":
        if not has_backends_arg:
            args.backends = ["cute-dsl"]
        if not has_input_dtype_arg:
            args.input_dtype = "bfloat16"
    if args.routine == "mm_fp8":
        if not has_backends_arg:
            args.backends = ["trtllm_low_latency"]
    if args.verbose >= 1:
        print(f"[INFO] {args = }")
    return args
