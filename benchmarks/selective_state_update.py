# Copyright (c) 2025, DeepLink.
import sys

import torch
import triton

import dlblas
from dlblas.kernels.selective_state_update import call as dlblas_selective_state_update

sys.path.append('..')
from tests.kernels.test_selective_state_update import selective_state_update_ref

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


def benchmark():
    device='cpu'
    rtol, atol = (5e-3, 3e-2)
    itype = torch.float16
    # set seed
    torch.random.manual_seed(0)
    dstate, ngroups, dim = 64, 2, 4096
    batch_size, headdim = 2, 64
    nheads = dim // headdim
    state = torch.randn(batch_size, nheads, headdim, dstate, dtype=itype, device=device)
    x = torch.randn(batch_size, nheads, headdim, device=device, dtype=itype)
    dt = torch.randn(batch_size, nheads, headdim, device=device, dtype=itype)
    dt_bias = torch.rand(nheads, headdim, device=device) - 4.0
    A = -torch.rand(nheads, headdim, dstate, device=device) - 1.0
    D = torch.randn(nheads, headdim, device=device)
    B = torch.randn(batch_size, ngroups, dstate, device=device)
    C = torch.randn(batch_size, ngroups, dstate, device=device)
    z = torch.randn_like(x)

    state_ref = state.detach().clone()

    # out = dlblas.selective_state_update(state, x, dt, A, B, C, D, z=z, dt_bias=dt_bias, dt_softplus=True)
    out = dlblas_selective_state_update(state, x, dt, A, B, C, D=D, z=z, dt_bias=dt_bias, dt_softplus=True)
    out_ref = selective_state_update_ref(state_ref, x, dt, A, B, C, D, z=z, dt_bias=dt_bias, dt_softplus=True)

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    assert torch.allclose(state, state_ref, rtol=rtol, atol=atol)
    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol)

    configs = []
    configs.append(
        triton.testing.Benchmark(
            x_names=['cnt'],
            x_vals=[1],
            line_arg='provider',
            line_vals=['triton', 'pytorch'],
            line_names=['Triton', 'PyTorch'],
            styles=[('red', '-'), ('blue', '-'), ('green', '-'), ('orange', '-')],
            ylabel='ms',
            plot_name=f"selective_state_update",
            args={},
        ))

    @triton.testing.perf_report(configs)
    def bench_fn(cnt, provider):
        warmup = 100
        rep = 100
        state_ref = state.detach().clone()
        if 'triton' in provider:
            fn = lambda: dlblas_selective_state_update(
                state_ref, x, dt, A, B, C, D, z=z, dt_bias=dt_bias, dt_softplus=True)
            # ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
            if device == 'cuda':
                ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
            else:
                print('input is cpu, use cpu_do_bench')
                ms = cpu_do_bench(fn, warmup=warmup, rep=rep)

        if 'pytorch' in provider:
            fn = lambda: selective_state_update_ref(
                state_ref, x, dt, A, B, C, D, z=z, dt_bias=dt_bias, dt_softplus=True)
            # ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
            if device == 'cuda':
                ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
            else:
                print('input is cpu, use cpu_do_bench')
                ms = cpu_do_bench(fn, warmup=warmup, rep=rep)
        return ms

    bench_fn.run(show_plots=True, print_data=True)


if __name__ == '__main__':
    benchmark()
    print('sucessfully!')
