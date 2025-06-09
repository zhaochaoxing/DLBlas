# Copyright (c) 2025, DeepLink.
# https://github.com/InternLM/lmdeploy/blob/v0.6.1/tests/pytorch/kernel/test_rms_norm.py
import pytest
import torch

from dlblas.kernels.rms_norm import rms_norm


class TestRMSNorm:

    @pytest.fixture(scope='class')
    def dtype(self, request):
        yield request.param

    @pytest.fixture(scope='class')
    def input(self, dtype):
        yield torch.rand(4, 8, dtype=dtype, device='npu')

    @pytest.fixture(scope='class')
    def weight(self, dtype):
        yield torch.rand(8, dtype=dtype, device='npu')

    @pytest.fixture(scope='class')
    def eps(self):
        yield 1e-6

    @pytest.fixture(scope='class')
    def gt(self, input, weight, eps):
        input_dtype = input.dtype
        input = input.to(torch.float32)
        variance = input.pow(2).mean(-1, keepdim=True)
        input = input * torch.rsqrt(variance + eps)
        return weight * input.to(input_dtype)

    @pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16, torch.float32], indirect=True)
    def test_rms_norm(self, input, weight, eps, gt):

        out = rms_norm(input, weight, eps)
        torch.testing.assert_close(out, gt)
