TRTLLM_RAGGED_ROW_ACTIVITY_MODES = (
    "assumed_active",
    "device_check",
    "cpu_mirror",
)

def normalize_backends(backends):
    """
    Normalize backend names planned for deprecation and print warnings.
    Currently:
    - Replaces deprecated 'trtllm-gen-native' with 'trtllm-native'.
    - Canonicalizes the Python-module spelling 'prims_ts' to 'prims-ts'.

    Args:
        backends: List of backend names

    Returns:
        List of normalized backend names
    """
    normalized = []
    for backend in backends:
        if backend == "trtllm-gen-native":
            print(
                "[WARNING] Backend name 'trtllm-gen-native' has been renamed to 'trtllm-native' and will be removed in a future release. "
            )
            normalized.append("trtllm-native")
        elif backend == "prims_ts":
            normalized.append("prims-ts")
        else:
            normalized.append(backend)
    return normalized

def parse_attention_args(line, parser):
    """
    Parse command line arguments for attention test configuration.

    Args:
        line: Command line arguments
        parser: ArgumentParser object already populated with shared arguments

    Returns:
        Parsed argument namespace
    """
    parser.add_argument(
        "--backends",
        type=str,
        required=False,
        nargs="+",
        default=None,
        choices=[
            "fa2",
            "fa2_tc",
            "fa3",
            "auto",
            "cudnn",
            "cudnn-native",
            "cutlass",
            "trtllm-gen",
            "trtllm-native",
            "trtllm-fmha-v2",
            "trtllm-gen-native",  # Deprecated, will be removed in future
            "cute-dsl",
            "prims-ts",
            "prims_ts",  # Accepted alias for the Python module spelling.
            "cute-dsl-prims",
        ],
        help="Kernel backends to test. Default: fa2, except DSV4 sparse MLA defaults to trtllm-gen. prims-ts selects the experimental task-scheduled Blackwell backend for the wrapper attention routines. backend=auto is supported for BatchDecodeWithPagedKVCacheWrapper, BatchPrefillWithPagedKVCacheWrapper, and BatchMLAPagedAttentionWrapper (where it pairs with --autotune to select between trtllm-gen and cute-dsl).",
    )
    parser.add_argument(
        "--page_size",
        type=int,
        required=False,
        default=0,
        help="Page size for paged attention. Required for paged attention. Ignored for non-paged attention.",
    )
    parser.add_argument(
        "--batch_size", type=int, required=True, help="Batch size of test case."
    )
    parser.add_argument(
        "--s_qo",
        type=int,
        required=False,
        default=1,
        help="Max sequence length of the query. For decode, 1 is standard decode and >1 enables speculative decode on supported backends.",
    )
    parser.add_argument(
        "--s_kv",
        type=int,
        required=True,
        help="Max sequence length of the key and value.",
    )
    parser.add_argument(
        "--num_qo_heads", type=int, required=True, help="Number of query heads."
    )
    parser.add_argument(
        "--num_kv_heads", type=int, required=True, help="Number of key and value heads."
    )
    parser.add_argument(
        "--head_dim_qk",
        type=int,
        required=False,
        help="Head dimension of the query and key for prefill and decode MHA/GQA/MQA.",
    )
    parser.add_argument(
        "--head_dim_vo",
        type=int,
        required=False,
        help="Head dimension of the value and output for prefill and decode MHA/GQA/MQ.",
    )
    parser.add_argument(
        "--head_dim_ckv",
        type=int,
        required=False,
        help="Head dimension of compressed kv-cache tensor (without rope).",
    )
    parser.add_argument(
        "--head_dim_kpe",
        type=int,
        required=False,
        help="Head dimension of the rope part of the kv-cache tensor.",
    )
    parser.add_argument(
        "--q_dtype",
        type=str,
        required=False,
        default="bfloat16",
        help="Query data type; supported values depend on the selected backend.",
    )
    parser.add_argument(
        "--kv_dtype",
        type=str,
        required=False,
        default="bfloat16",
        help="Key/value data type; supported values depend on the selected backend.",
    )
    parser.add_argument(
        "--v_dtype",
        type=str,
        required=False,
        default=None,
        help="Value data type. If not specified, defaults to kv_dtype. Only the "
        "prims-ts backend accepts a V dtype that differs from kv_dtype "
        "(QK-BF16/PV-FP8).",
    )
    parser.add_argument(
        "--out_dtype",
        type=str,
        required=False,
        default=None,
        help="Data type of the output. If not specified, defaults to q_dtype.",
    )
    parser.add_argument(
        "--causal",
        action="store_true",
        default=False,
        help="Enable bottom-right causal masking for backends that support it.",
    )
    parser.add_argument(
        "--spec_dec_mask",
        type=str,
        choices=["causal", "full"],
        default="causal",
        help=(
            "Draft-block mask for speculative decode (--s_qo > 1) in decode tests. "
            "'causal': draft token i attends to draft tokens j <= i (standard "
            "speculative decoding, the default). 'full': every draft token attends "
            "to all draft tokens (e.g. DFlash-style drafters). The KV prefix is "
            "always fully visible."
        ),
    )
    parser.add_argument(
        "--random_actual_seq_len",
        action="store_true",
        default=False,
        help="Use random actual sequence lengths for the query and key and value. Random values are generated between 1 and maximum sequence length. If False, use maximum sequence length.",
    )
    parser.add_argument(
        "--row_activity_mode",
        choices=TRTLLM_RAGGED_ROW_ACTIVITY_MODES,
        default=None,
        help=(
            "TRTLLM native ragged prefill only: choose the all-active fast path, "
            "device-side row check, or CPU sequence-length mirrors."
        ),
    )
    parser.add_argument(
        "--calls_per_sample",
        type=int,
        default=1,
        help="TRTLLM native ragged prefill only: calls grouped into each timing sample.",
    )
    parser.add_argument(
        "--autotune",
        action="store_true",
        default=False,
        help=(
            "Enable autotuner warmup for supported attention routines "
            "(BatchMLAPagedAttentionWrapper with trtllm-native / cute-dsl). "
            "Pre-tunes the kernel configuration before timing so the steady-state "
            "measurement reflects the autotuned tactic."
        ),
    )
    parser.add_argument(
        "--mla_is_var_seq",
        choices=["true", "false", "auto"],
        default=None,
        help=(
            "MLA-only: control the is_var_seq argument passed to "
            "trtllm_batch_decode_with_kv_cache_mla, which selects the var-seq vs. "
            "persistent scheduler (is_persistent = not is_var_seq). "
            "'true'/'false' force the value; 'auto' resolves to --random_actual_seq_len. "
            "If unset (default), is_var_seq is not passed and the API default (True) "
            "is used, preserving existing behavior and perf baselines."
        ),
    )
    parser.add_argument(
        "--mla_cute_dsl_impl",
        choices=["auto", "modular", "monolithic"],
        default=None,
        help=(
            "MLA-only: control the cute_dsl_impl argument passed to "
            "trtllm_batch_decode_with_kv_cache_mla, selecting the CuTe DSL "
            "decode implementation. 'auto' (API default) runs monolithic and "
            "only promotes to modular for modular-only features (e.g. sinks); "
            "'modular'/'monolithic' force that impl. If unset (default), "
            "cute_dsl_impl is not passed and the API default ('auto') is used, "
            "preserving existing behavior and perf baselines."
        ),
    )
    parser.add_argument(
        "--mla_softmax_rescale_threshold_log2",
        type=float,
        default=0.0,
        help=(
            "BatchMLAPagedAttentionWrapper routine with the prims-ts backend only "
            "(ignored by other routines): passed to BatchMLADecodePagedTSWrapper."
            "plan(tuning=MLAAttentionTuning(softmax_rescale_threshold_log2=...)). "
            "A positive value skips the "
            "O rescale while a K tile raises a row maximum by at most this many "
            "binades (bound: 8 for FP8, 32 for BF16); FP8 P is then quantized at "
            "448 * 2**-T instead of 448. 0 (default) keeps the exact rescale."
        ),
    )
    parser.add_argument(
        "--swa_topk",
        type=int,
        default=128,
        help="DSV4 sparse MLA only: fixed sliding-window segment width.",
    )
    parser.add_argument(
        "--compressed_topk",
        type=int,
        default=1920,
        help="DSV4 sparse MLA only: maximum selected compressed-cache rows per query.",
    )
    parser.add_argument(
        "--compressed_kv_len",
        type=int,
        default=None,
        help=(
            "DSV4 sparse MLA only: logical compressed-cache rows per request "
            "(default: ceil(s_kv / 4))."
        ),
    )
    parser.add_argument(
        "--compressed_page_size",
        type=int,
        default=64,
        help="DSV4 sparse MLA only: compressed-cache page size.",
    )
    parser.add_argument(
        "--kv_layout",
        choices=["HND", "NHD"],
        default="HND",
        help="DSV4 sparse MLA only: layout of both KV-cache pools.",
    )

    args = parser.parse_args(line)

    if args.backends is None:
        args.backends = (
            ["trtllm-gen"]
            if args.routine == "trtllm_batch_decode_sparse_mla_dsv4"
            else ["fa2"]
        )

    if args.routine == "trtllm_batch_decode_sparse_mla_dsv4" and args.page_size == 0:
        args.page_size = 256
    if args.routine == "trtllm_batch_decode_sparse_mla_dsv4":
        # The sparse tables encode causal visibility; there is no non-causal
        # mode in this DSV4 generation-form API.
        args.causal = True

    # Normalize backend names (handle deprecated names)
    args.backends = normalize_backends(args.backends)
    if args.calls_per_sample < 1:
        raise ValueError("--calls_per_sample must be positive")
    if args.row_activity_mode is None and args.calls_per_sample != 1:
        raise ValueError("--calls_per_sample requires --row_activity_mode")
    if args.row_activity_mode is not None:
        if args.routine != "BatchPrefillWithRaggedKVCacheWrapper":
            raise ValueError("--row_activity_mode only supports ragged prefill")
        if args.backends != ["trtllm-native"]:
            raise ValueError("--row_activity_mode requires only trtllm-native")
        if not args.no_cuda_graph or not args.use_cuda_events:
            raise ValueError(
                "--row_activity_mode requires --no_cuda_graph and --use_cuda_events"
            )
    if args.verbose >= 1:
        print(f"[INFO] {args = }")
    return args
