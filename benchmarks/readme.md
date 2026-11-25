# Benchmark Tool for FlashInfer-Ascend

## Usage

- GEMM

```
python3 -m benchmarks.flashinfer_benchmark --routine mm_bf16  --output_path .log/gemm.mm_bf16.csv --m 8 --n 128 --k 128

python3 -m benchmarks.flashinfer_benchmark --routine mm_fp4 --output_path .log/gemm.mm_fp4.csv --m 8 --n 128 --k 128
```