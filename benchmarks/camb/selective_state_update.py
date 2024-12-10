import sys
import torch
import torch_mlu
from torch_mlu.utils.model_transfer import transfer
import triton
import dlblas
import os
sys.path.append("/workspace/volume/shangda/linming/dlBLAS/tests/kernels")
print(sys.path)
from test_selective_state_update import selective_state_update_ref


def check_output(o1, o2, reduce_dim = 1):
    eps = 1e-5
    print('diff mean', (o1 - o2).abs().mean())
    print('diff abs max ', (o1 - o2).abs().max())
    print('relative diff max ', ((o1 - o2).abs() / ( o2.abs() + eps)).max())

    assert torch.allclose(o1, o2, atol=1e-4 * reduce_dim)
    assert torch.allclose(o1, o2, rtol=5e-3)

def benchmark():
    device = "mlu"
    rtol, atol =  (5e-3, 3e-2)
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
    
    from dlblas.kernels.camb import selective_state_update
    out = selective_state_update.call(state, x, dt, A, B, C, D, z=z, dt_bias=dt_bias, dt_softplus=True)
    out_ref = selective_state_update_ref(state_ref, x, dt, A, B, C, D, z=z, dt_bias=dt_bias, dt_softplus=True)

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    check_output(out, out_ref)

    configs = []
    configs.append(
        triton.testing.Benchmark(      
            x_names=["cnt"],
            x_vals=[1],

            line_arg="provider",
            line_vals=["triton", "pytorch"],
            line_names=["Triton", "PyTorch"],

            styles=[("red", "-"), ("blue", "-"), ("green", "-"), ("orange", "-")],
            ylabel="ms",
            plot_name=f"selective_state_update",
            args={
            },
        ))
    @triton.testing.perf_report(configs)
    def bench_fn(cnt, provider):
        warmup = 100
        rep = 100
        state_ref = state.detach().clone()
        if "triton" in provider:
            fn = lambda: dlblas.selective_state_update(state_ref, x, dt, A, B, C, D, z=z, dt_bias=dt_bias, dt_softplus=True)
            ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)

        if "pytorch" in provider:
            fn = lambda : selective_state_update_ref(state_ref, x, dt, A, B, C, D, z=z, dt_bias=dt_bias, dt_softplus=True)
            ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        return ms
    
    bench_fn.run(show_plots=True, print_data=True)

if __name__ == '__main__':
    benchmark()
    print("sucessfully!")
