import triton
import torch
import torch_mlu
from torch_mlu.utils.model_transfer import transfer
import dlblas
# from python.dlBLAS.dlblas.utils.device_utils import get_idle_device
from test_utils import test_latency_and_output

def _rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def torch_rotary_pos_emb(q_states, k_states, cached_cos, cached_sin, position_ids_1d):
    cos = cached_cos[position_ids_1d, None, :]
    sin = cached_sin[position_ids_1d, None, :]
    q_embed = q_states * cos + _rotate_half(q_states) * sin
    k_embed = k_states * cos + _rotate_half(k_states) * sin
    return q_embed, k_embed

eps = 1e-5

def test():
    # device_ = torch.device(get_idle_device())
    # torch.cuda.set_device(device_)
    device_ = "mlu"
    print(device_)
    dtype_ = torch.float16
    b, s, h, d = 1, 256, 32, 128
    cached_cos = torch.randn((s, d), dtype=dtype_, device=device_)
    cached_sin = torch.randn((s, d), dtype=dtype_, device=device_)
    q_states = torch.randn((b * s, h, d), dtype=dtype_, device=device_)
    k_states = torch.randn((b * s, h, d), dtype=dtype_, device=device_)
    # position_ids_1d = torch.randint(0, s, (b * s,), device=device_)
    position_ids_1d = torch.arange(0, s, device=device_)
    
    q_embed, k_embed = torch_rotary_pos_emb(
        q_states, k_states, cached_cos, cached_sin, position_ids_1d
    )
    from dlblas.kernels.camb.apply_rotary_pos_emb import apply_rotary_pos_emb
    q_embed_tri, k_embed_tri = apply_rotary_pos_emb(
        q_states, k_states, cached_cos, cached_sin
    )
    out, speed = test_latency_and_output(apply_rotary_pos_emb, (q_states, k_states, cached_cos, cached_sin))
    print("cost", speed * 1000, "ms")
    print("max abs diff: ", torch.max(abs(q_embed - q_embed_tri)))
    print('relative diff max ', ((q_embed - q_embed_tri).abs() / (q_embed.abs() + eps) ).max())
    print("max abs diff: ", torch.max(abs(k_embed - k_embed_tri)))
    print('relative diff max ', ((k_embed - k_embed_tri).abs() / (k_embed.abs() + eps) ).max())
    assert torch.allclose(q_embed, q_embed_tri, atol=1e-4, rtol=5e-3)
    assert torch.allclose(k_embed, k_embed_tri, atol=1e-4, rtol=5e-3)


if __name__ == "__main__":

    test()
    print("sucessfully!")
