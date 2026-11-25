# Benchmark Tool for FlashInfer-Ascend

## Usage

- GEMM

```
python3 -m benchmarks.flashinfer_benchmark --routine mm_bf16  --output_path .log/gemm.mm_bf16.csv --m 8 --n 128 --k 128 --backend auto

python3 -m benchmarks.flashinfer_benchmark --routine bmm_fp8 --output_path .log/gemm.bmm_fp8.csv --m 128 --n 128 --k 128 --backend auto
```