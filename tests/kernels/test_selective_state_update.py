# Copyright (c) 2025, DeepLink.
import pytest
import torch
import torch.nn.functional as F
from einops import rearrange, repeat

import dlblas


def selective_state_update_ref(state, x, dt, A, B, C, D=None, z=None, dt_bias=None, dt_softplus=False):
    """
    Argument:
        state: (batch, dim, dstate) or (batch, nheads, dim, dstate)
        x: (batch, dim) or (batch, nheads, dim)
        dt: (batch, dim) or (batch, nheads, dim)
        A: (dim, dstate) or (nheads, dim, dstate)
        B: (batch, dstate) or (batch, ngroups, dstate)
        C: (batch, dstate) or (batch, ngroups, dstate)
        D: (dim,) or (nheads, dim)
        z: (batch, dim) or (batch, nheads, dim)
        dt_bias: (dim,) or (nheads, dim)
    Return:
        out: (batch, dim) or (batch, nheads, dim)
    """
    has_heads = state.dim() > 3
    if state.dim() == 3:
        state = state.unsqueeze(1)
    if x.dim() == 2:
        x = x.unsqueeze(1)
    if dt.dim() == 2:
        dt = dt.unsqueeze(1)
    if A.dim() == 2:
        A = A.unsqueeze(0)
    if B.dim() == 2:
        B = B.unsqueeze(1)
    if C.dim() == 2:
        C = C.unsqueeze(1)
    if D is not None and D.dim() == 1:
        D = D.unsqueeze(0)
    if z is not None and z.dim() == 2:
        z = z.unsqueeze(1)
    if dt_bias is not None and dt_bias.dim() == 1:
        dt_bias = dt_bias.unsqueeze(0)
    batch, nheads, dim, dstate = state.shape
    assert x.shape == (batch, nheads, dim)
    assert dt.shape == x.shape
    assert A.shape == (nheads, dim, dstate)
    ngroups = B.shape[1]
    assert nheads % ngroups == 0, 'nheads must be divisible by ngroups'
    assert B.shape == (batch, ngroups, dstate)
    assert C.shape == B.shape
    if D is not None:
        assert D.shape == (nheads, dim)
    if z is not None:
        assert z.shape == x.shape
    if dt_bias is not None:
        assert dt_bias.shape == (nheads, dim)
        dt = dt + dt_bias
    dt = F.softplus(dt) if dt_softplus else dt
    dA = torch.exp(rearrange(dt, 'b h d -> b h d 1') * A)  # (batch, nheads, dim, dstate)
    B = repeat(B, 'b g n -> b (g h) n', h=nheads // ngroups)  # (batch, nheads, dstate)
    C = repeat(C, 'b g n -> b (g h) n', h=nheads // ngroups)  # (batch, nheads, dstate)
    dB = rearrange(dt, 'b h d -> b h d 1') * rearrange(B, 'b h n -> b h 1 n')  # (batch, nheads, dim, dstate)
    state.copy_(state * dA + dB * rearrange(x, 'b h d -> b h d 1'))  # (batch, dim, dstate
    out = torch.einsum('bhdn,bhn->bhd', state.to(C.dtype), C)
    if D is not None:
        out += (x * D).to(out.dtype)
    out = (out if z is None else out * F.silu(z)).to(x.dtype)
    if not has_heads:
        out = out.squeeze(1)
    return out


@pytest.mark.parametrize('itype', [torch.float32, torch.float16, torch.bfloat16])
# @pytest.mark.parametrize('itype', [torch.float16])
@pytest.mark.parametrize('has_z', [False, True])
# @pytest.mark.parametrize('has_z', [True])
@pytest.mark.parametrize('dstate', [16, 32, 64])
# @pytest.mark.parametrize("dstate", [16])
@pytest.mark.parametrize('dim', [2048, 2048 + 16, 4096])
# @pytest.mark.parametrize("dim", [2048])
def test_selective_state_update(dim, dstate, has_z, itype):
    device = 'npu'
    rtol, atol = (3e-4, 1e-3) if itype == torch.float32 else (5e-3, 1e-2)
    if itype == torch.bfloat16:
        rtol, atol = 1e-2, 5e-2
        if torch.version.hip:
            atol *= 2
    # set seed
    torch.random.manual_seed(0)
    batch_size = 2
    state = torch.randn(batch_size, dim, dstate, dtype=itype, device=device)
    x = torch.randn(batch_size, dim, device=device, dtype=itype)
    dt = torch.randn(batch_size, dim, device=device, dtype=itype)
    dt_bias = torch.rand(dim, device=device) - 4.0
    A = -torch.rand(dim, dstate, device=device) - 1.0
    B = torch.randn(batch_size, dstate, device=device)
    C = torch.randn(batch_size, dstate, device=device)
    D = torch.randn(dim, device=device)
    if has_z:
        z = torch.randn_like(x)
    else:
        z = None
    state_ref = state.detach().clone()
    out = dlblas.selective_state_update(state, x, dt, A, B, C, D=D, z=z, dt_bias=dt_bias, dt_softplus=True)
    out_ref = selective_state_update_ref(state_ref, x, dt, A, B, C, D=D, z=z, dt_bias=dt_bias, dt_softplus=True)

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    assert torch.allclose(state, state_ref, rtol=rtol, atol=atol)
    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize('itype', [torch.float32, torch.float16, torch.bfloat16])
# @pytest.mark.parametrize('itype', [torch.float32])
# @pytest.mark.parametrize("has_z", [False, True])
@pytest.mark.parametrize('has_z', [False])
# @pytest.mark.parametrize("tie_hdim", [False, True])
@pytest.mark.parametrize('tie_hdim', [False])
# @pytest.mark.parametrize("ngroups", [1, 2, 4])
@pytest.mark.parametrize('ngroups', [2])
# @pytest.mark.parametrize("dstate", [16, 32, 64])
@pytest.mark.parametrize('dstate', [16])
# @pytest.mark.parametrize("dim", [2048, 4096])
@pytest.mark.parametrize('dim', [2048])
def test_selective_state_update_with_heads(dim, dstate, ngroups, has_z, tie_hdim, itype):
    device = 'npu'
    rtol, atol = (3e-4, 1e-3) if itype == torch.float32 else (5e-3, 3e-2)
    if itype == torch.bfloat16:
        rtol, atol = 1e-2, 1e-1
    # set seed
    torch.random.manual_seed(0)
    batch_size = 2
    headdim = 64
    nheads = dim // headdim
    state = torch.randn(batch_size, nheads, headdim, dstate, dtype=itype, device=device)
    x = torch.randn(batch_size, nheads, headdim, device=device, dtype=itype)
    if not tie_hdim:
        dt = torch.randn(batch_size, nheads, headdim, device=device, dtype=itype)
        dt_bias = torch.rand(nheads, headdim, device=device) - 4.0
        A = -torch.rand(nheads, headdim, dstate, device=device) - 1.0
        D = torch.randn(nheads, headdim, device=device)
    else:
        dt = repeat(torch.randn(batch_size, nheads, device=device, dtype=itype), 'b h -> b h p', p=headdim)
        dt_bias = repeat(torch.rand(nheads, device=device) - 4.0, 'h -> h p', p=headdim)
        A = repeat(-torch.rand(nheads, device=device) - 1.0, 'h -> h p n', p=headdim, n=dstate)
        D = repeat(torch.randn(nheads, device=device), 'h -> h p', p=headdim)
    B = torch.randn(batch_size, ngroups, dstate, device=device)
    C = torch.randn(batch_size, ngroups, dstate, device=device)
    if has_z:
        z = torch.randn_like(x)
    else:
        z = None
    state_ref = state.detach().clone()
    out = dlblas.selective_state_update(state, x, dt, A, B, C, D, z=None, dt_bias=dt_bias, dt_softplus=True)
    out_ref = selective_state_update_ref(state_ref, x, dt, A, B, C, D, z=None, dt_bias=dt_bias, dt_softplus=True)

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    assert torch.allclose(state, state_ref, rtol=rtol, atol=atol)
    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol)
