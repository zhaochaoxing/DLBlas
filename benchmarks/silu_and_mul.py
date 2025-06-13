# Copyright (c) 2025, DeepLink.
# https://github.com/InternLM/lmdeploy/blob/v0.6.1/tests/pytorch/kernel/test_activation.py
import pytest
# import torch_mlu
# import torch_mlu.utils.gpu_migration
import torch
import triton

from dlblas.kernels.activation import silu_and_mul

import time
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

class TestSiluAndMul:

    @pytest.fixture
    def seqlen(self):
        yield 256

    @pytest.fixture
    def feat_size(self, request):
        yield request.param

    @pytest.fixture
    def x(self, seqlen, feat_size):
        yield torch.rand(seqlen, feat_size, dtype=torch.float16, device='cpu')

    @pytest.fixture
    def gt(self, x):
        gate, up = x.chunk(2, -1)
        gate = torch.nn.functional.silu(gate)
        yield gate * up

    @pytest.mark.parametrize('feat_size', [4096, 768], indirect=True)
    def test_silu_and_mul(self, x, gt):
        out = silu_and_mul(x)
        torch.testing.assert_close(out, gt)


def _gt(x):
    gate, up = x.chunk(2, -1)
    gate = torch.nn.functional.silu(gate)
    return gate * up


def _test_silu_and_mul(x):
    return silu_and_mul(x)


def test():
    seqlen = 256
    feat_size = 4096
    x = torch.rand(seqlen, feat_size, dtype=torch.float16, device='cpu')

    gt = _gt(x)
    tt = _test_silu_and_mul(x)

    print('max diff', (gt - tt).abs().max())

    configs = []
    configs.append(
        triton.testing.Benchmark(
            x_names=['op'],
            x_vals=['fwd'],
            line_arg='provider',
            line_vals=['triton', 'pytorch'],
            line_names=['Triton', 'PyTorch'],
            ylabel='ms',
            plot_name='',
            args={},
        ))

    @triton.testing.perf_report(configs)
    def bench_fn(op, provider, device='cpu'):
        warmup = 100
        rep = 200

        if 'triton' in provider:
            # fn = lambda: test_paged_attention(conti_q, blocked_kv, block_offsets, start_loc, seq_lens, history_lens, feat_dim_v)
            fn = lambda: silu_and_mul(x)
        if 'pytorch' in provider:
            fn = lambda: _gt(x)

        # ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        
        if device == 'cuda':
            ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        else:
            print('input is cpu, use cpu_do_bench')
            ms = cpu_do_bench(fn, warmup=warmup, rep=rep)
        return ms

    bench_fn.run(show_plots=True, print_data=True)


if __name__ == '__main__':
    test()
