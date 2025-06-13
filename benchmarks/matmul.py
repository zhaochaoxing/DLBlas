# Copyright (c) 2025, DeepLink.
import argparse
import os

import torch
import triton
import triton.language as tl

from dlblas.utils import get_op


DEVICE = 'cpu'
TEST_CPU = True
def change_env():
    global DEVICE
    if TEST_CPU:
        from triton.backends.triton_shared.driver import CPUDriver

        def select_cpu_backend():
            triton.runtime.driver.set_active(CPUDriver())

        select_cpu_backend()
        DEVICE = 'cpu'
    else:
        from dlblas.utils.device_utils import get_idle_device
        DEVICE = torch.device(get_idle_device())
        torch.cuda.set_device(DEVICE)

    print(f"zmz debug device={triton.runtime.driver.active.get_current_target()}, DEVICE={DEVICE}")

change_env()


import time

def cpu_do_bench(fn, warmup=25, rep=100):
    # 预热
    for _ in range(warmup):
        fn()
    
    # 实际测量
    start_time = time.perf_counter()
    for _ in range(rep):
        fn()
    end_time = time.perf_counter()
    
    # 计算平均时间 (毫秒)
    return (end_time - start_time) * 1000 / rep



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=int, default=32)
    parser.add_argument('-n', type=int, default=32)
    parser.add_argument('-k', type=int, default=16)
    parser.add_argument('--bench', default=False, action=argparse.BooleanOptionalAction)

    return parser.parse_args()


def is_cuda():
    # return torch.cuda.is_available()
    return False


def main():
    args = parse_args()
    dtype = torch.float16
    device = 'cpu'
    a = torch.randn(
        (args.m, args.k),
        dtype=dtype,
        device=device,
    )
    b = torch.randn(
        (args.k, args.n),
        dtype=dtype,
        device=device,
    )
    matmul = get_op('matmul', (a, b))
    # test
    out = matmul(a, b)
    ref_out = a @ b
    tol = {
        'atol': 1.0,
    }
    if torch.allclose(out, ref_out, **tol):
        print('✅ Triton and Torch match')
    else:
        print('❌ Triton and Torch differ')

    # skip benchmarks
    if not args.bench:
        return

    # TORCH_HAS_FP8 = torch.cuda.is_available() and (torch.cuda.get_device_capability()[0] >= 8)
    TORCH_HAS_FP8 = False
    ref_lib = 'cuBLAS'
    configs = []
    for fp8_inputs in [False, True]:
        if fp8_inputs and (not TORCH_HAS_FP8 or not is_cuda()):
            continue
        configs.append(
            triton.testing.Benchmark(
                x_names=['cnt'],  # Argument names to use as an x-axis for the plot
                # x_vals=[128 * i for i in range(10, 15)],  # Different possible values for `x_name`
                x_vals=[1],  # NOTE: the tunning framework specialized to one shape
                line_arg='provider',  # Argument name whose value corresponds to a different line in the plot
                # Possible values for `line_arg`
                # Don't compare to cublas for fp8 cases as torch.matmul doesn't support fp8 at the moment.
                line_vals=['triton'] if fp8_inputs else [ref_lib.lower(), 'triton'],  # Label name for the lines
                line_names=['Triton'] if fp8_inputs else [ref_lib, 'Triton'],  # Line styles
                styles=[('green', '-'), ('blue', '-')],
                ylabel='TFLOPS',  # Label name for the y-axis
                plot_name='matmul-performance-' +
                ('fp16'
                 if not fp8_inputs else 'fp8'),  # Name for the plot, used also as a file name for saving the plot.
                args={
                    'fp8_inputs': fp8_inputs,
                    'M': args.m,
                    'N': args.n,
                    'K': args.k,
                },
            ))

    @triton.testing.perf_report(configs)
    def benchmark(cnt, M, N, K, provider, fp8_inputs):
        warmup = 500
        rep = 500
        a = torch.randn((M, K), device='cpu', dtype=torch.float16)
        b = torch.randn((K, N), device='cpu', dtype=torch.float16)
        if TORCH_HAS_FP8 and fp8_inputs:
            a = a.to(torch.float8_e5m2)
            b = b.T
            b = b.to(torch.float8_e5m2)
        quantiles = [0.5, 0.2, 0.8]
        if provider == ref_lib.lower():
            ms, min_ms, max_ms = cpu_do_bench(lambda: torch.matmul(a, b),
                                                         quantiles=quantiles,
                                                         warmup=warmup,
                                                         rep=rep)
        if provider == 'triton':
            ms, min_ms, max_ms = cpu_do_bench(lambda: matmul(a, b),
                                                         quantiles=quantiles,
                                                         warmup=warmup,
                                                         rep=rep)
        perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
        return perf(ms), perf(max_ms), perf(min_ms)

    benchmark.run(show_plots=False, print_data=True)


if __name__ == '__main__':
    main()
