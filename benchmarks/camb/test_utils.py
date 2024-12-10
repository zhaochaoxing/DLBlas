import time
import torch
import torch_mlu
@torch.no_grad()
def test_latency_and_output(ops, input, warmup=50, iteration=1000):
    output = ops(*input)
    # output = [o.clone() for o in output]

    for _ in range(warmup):
        _ = ops(*input)
        torch.mlu.synchronize()

    time1 = time.time()
    for _ in range(iteration):
        _ = ops(*input)
        torch.mlu.synchronize()
    time2 = time.time()
    return output, (time2 - time1) / iteration

eps = 1e-5

# o2: ref result
def check_output(o1, o2, reduce_dim = 1):
    print('diff mean', (o1 - o2).abs().mean())
    print('diff abs max ', (o1 - o2).abs().max())
    print('relative diff max ', ((o1 - o2).abs() / ( o2.abs() + eps)).max())

    assert torch.allclose(o1, o2, atol=1e-4 * reduce_dim)
    assert torch.allclose(o1, o2, rtol=5e-3)
