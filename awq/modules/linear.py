import math
import torch
import numpy as np
import torch.nn as nn
import awq_ext  # with CUDA kernels

# pip install git+https://github.com/IST-DASLab/marlin.git
import marlin_cuda

def make_divisible(c, divisor):
    return (c + divisor - 1) // divisor

def calculate_zeros_width(in_features, group_size=128, pack_num=8):
    if group_size >= 128:
        size_multiplier = 1
    elif group_size == 64:
        size_multiplier = 2
    elif group_size == 32:
        size_multiplier = 4
    else:
        raise NotImplementedError
    
    base_width = make_divisible(in_features // group_size, pack_num)
    base_width = make_divisible(base_width, size_multiplier) * size_multiplier
    return base_width

class WQLinear_GEMM(nn.Module):
    def __init__(self, w_bit, group_size, in_features, out_features, bias, dev):
        super().__init__()
        
        if w_bit not in [4]:
            raise NotImplementedError("Only 4-bit are supported for now.")
        
        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.group_size = group_size if group_size != -1 else in_features
        
        # quick sanity check (make sure aligment)
        assert self.in_features % self.group_size == 0
        assert out_features % (32 // self.w_bit) == 0

        self.register_buffer('qweight', torch.zeros((in_features, out_features // (32 // self.w_bit)), dtype=torch.int32, device=dev))
        self.register_buffer('qzeros', torch.zeros((in_features // self.group_size, out_features // (32 // self.w_bit)), dtype=torch.int32, device=dev))
        self.register_buffer('scales', torch.zeros((in_features // self.group_size, out_features), dtype=torch.float16, device=dev))
        if bias:
            self.register_buffer('bias', torch.zeros((out_features), dtype=torch.float16, device=dev))
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, linear, w_bit, group_size, init_only=False, scales=None, zeros=None):
        awq_linear = cls(w_bit, group_size, linear.in_features, linear.out_features, linear.bias is not None, linear.weight.device)
        if init_only:  # just prepare for loading sd
            return awq_linear
        
        # need scales and zeros info for real quantization
        assert scales is not None and zeros is not None  
        scale_zeros = zeros * scales
        
        awq_linear.scales = scales.clone().half()
        if linear.bias is not None:
            awq_linear.bias = linear.bias.clone().half()

        pack_num = 32 // awq_linear.w_bit
        
        intweight = []
        for idx in range(awq_linear.in_features):
            intweight.append(torch.round((linear.weight.data[:, idx] + scale_zeros[idx // group_size]) / awq_linear.scales[idx // group_size]).to(torch.int)[:, None])
        intweight = torch.cat(intweight, dim=1)
        intweight = intweight.t().contiguous()
        intweight = intweight.to(dtype=torch.int32)
        qweight = torch.zeros((intweight.shape[0], intweight.shape[1] // 32 * awq_linear.w_bit), dtype=torch.int32, device=intweight.device)           
         
        for col in range(intweight.shape[1] // pack_num):
            if awq_linear.w_bit == 4:
                order_map = [0, 2, 4, 6, 1, 3, 5, 7]
            else:
                raise NotImplementedError("Only 4-bit are supported for now.")
            for i in range(pack_num):
                qweight_col = intweight[:, col * pack_num + order_map[i]]
                qweight[:, col] |= qweight_col << (i * awq_linear.w_bit)
        awq_linear.qweight = qweight

        zeros = zeros.to(dtype=torch.int32)
        qzeros = torch.zeros((zeros.shape[0], zeros.shape[1] // 32 * awq_linear.w_bit), dtype=torch.int32, device=zeros.device)
        
        for col in range(zeros.shape[1] // pack_num):
            if awq_linear.w_bit == 4:
                order_map = [0, 2, 4, 6, 1, 3, 5, 7]
            else:
                raise NotImplementedError("Only 4-bit are supported for now.")
            for i in range(pack_num):
                qzero_col = zeros[:, col * pack_num + order_map[i]]
                qzeros[:, col] |= qzero_col << (i * awq_linear.w_bit)
        awq_linear.qzeros = qzeros
        
        return awq_linear

    @torch.no_grad()
    def forward(self, x):
        out_shape = x.shape[:-1] + (self.out_features, )

        input_dtype = x.dtype
        if input_dtype != torch.float16:
            x = x.half()
        
        out = awq_ext.gemm_forward_cuda(x.reshape(-1, x.shape[-1]), self.qweight, self.scales, self.qzeros, 8)
        
        if input_dtype != torch.float16:
            out = out.to(dtype=input_dtype)
        
        out = out + self.bias if self.bias is not None else out
        return out.reshape(out_shape)
    
    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}, w_bit={}, group_size={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.w_bit, self.group_size
        )


class WQLinear_GEMV(nn.Module):
    def __init__(self, w_bit, group_size, in_features, out_features, bias, dev):
        super().__init__()
        
        if w_bit not in [4]:
            raise NotImplementedError("Only 4-bit are supported for now.")
        
        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.group_size = group_size if group_size != -1 else in_features
        self.split_k_iters = 8

        # quick sanity check (make sure aligment)
        assert self.in_features % self.group_size == 0
        assert out_features % (32 // self.w_bit) == 0
        pack_num = (32 // self.w_bit)

        self.register_buffer('qweight', torch.zeros((out_features, in_features // pack_num), dtype=torch.int32, device=dev))
        self.register_buffer('qzeros', torch.zeros((out_features, calculate_zeros_width(in_features, self.group_size)), dtype=torch.int32, device=dev))
        self.register_buffer('scales', torch.zeros((out_features, calculate_zeros_width(in_features, self.group_size) * pack_num), dtype=torch.float16, device=dev))
        if bias:
            self.register_buffer('bias', torch.zeros((out_features), dtype=torch.float16, device=dev))
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, linear, w_bit, group_size, init_only=False, scales=None, zeros=None):
        awq_linear = cls(w_bit, group_size, linear.in_features, linear.out_features, linear.bias is not None, linear.weight.device)
        if init_only:  # just prepare for loading sd
            return awq_linear
        
        # need scales and zeros info for real quantization
        assert scales is not None and zeros is not None  
        scale_zeros = zeros * scales

        pack_num = 32 // awq_linear.w_bit
        qscales = torch.zeros(
            (scales.shape[0], calculate_zeros_width(linear.in_features, group_size) * pack_num),
            dtype=torch.float16,
            device=scales.device
        )
        qscales[:, :scales.shape[1]] = scales
        awq_linear.scales = qscales
        if linear.bias is not None:
            awq_linear.bias = linear.bias.clone().half()
        
        intweight = []
        for idx in range(awq_linear.in_features):
            intweight.append(torch.round((linear.weight.data[:, idx] + scale_zeros[:, idx // group_size]) / awq_linear.scales[:, idx // group_size]).to(torch.int)[:, None])
        intweight = torch.cat(intweight, dim=1)
        intweight = intweight.to(dtype=torch.int32)
        qweight = torch.zeros((intweight.shape[0], intweight.shape[1] // 32 * awq_linear.w_bit), dtype=torch.int32, device=intweight.device)           
         
        for col in range(intweight.shape[1] // pack_num):
            if awq_linear.w_bit == 4:
                order_map = [0, 1, 2, 3, 4, 5, 6, 7]
            else:
                raise NotImplementedError("Only 4-bit are supported for now.")
            for i in range(pack_num):
                qweight_col = intweight[:, col * pack_num + order_map[i]]
                qweight[:, col] |= qweight_col << (i * awq_linear.w_bit)
        awq_linear.qweight = qweight

        zeros = zeros.to(dtype=torch.int32)
        qzeros = torch.zeros(
            (zeros.shape[0], calculate_zeros_width(linear.in_features, group_size)),
            dtype=torch.int32,
            device=zeros.device,
        )
        
        for col in range((zeros.shape[1] + pack_num - 1) // pack_num):
            if awq_linear.w_bit == 4:
                order_map = [0, 1, 2, 3, 4, 5, 6, 7]
            else:
                raise NotImplementedError("Only 4-bit are supported for now.")
            for i in range(pack_num):
                if col * pack_num + order_map[i] >= zeros.shape[1]:
                    continue
                qzero_col = zeros[:, col * pack_num + order_map[i]]
                qzeros[:, col] |= qzero_col << (i * awq_linear.w_bit)
        awq_linear.qzeros = qzeros
        return awq_linear

    @torch.no_grad()
    def forward(self, x):
        out_shape = x.shape[:-1] + (self.out_features, )
        inputs = x.reshape(-1, x.shape[-1])

        input_dtype = inputs.dtype
        if input_dtype != torch.float16:
            inputs = inputs.half()
        
        if inputs.shape[0] > 8:
            out = awq_ext.gemmv2_forward_cuda(inputs, self.qweight, self.scales, self.qzeros, self.group_size, self.split_k_iters)
        else:
            out = awq_ext.gemv_forward_cuda(inputs, self.qweight, self.scales, self.qzeros, self.group_size)

        if input_dtype != torch.float16:
            out = out.to(dtype=input_dtype)
        
        out = out + self.bias if self.bias is not None else out
        return out.reshape(out_shape)
    
    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}, w_bit={}, group_size={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.w_bit, self.group_size
        )


def mul(A, B, C, s, workspace, thread_k=-1, thread_n=-1, sms=-1):
    """Marlin FP16xINT4 multiply; can be used within `torch.compile`.
    @A: `torch.half` input matrix of shape `(m, k)` in standard row-major layout
    @B: `torch.int` weight matrix of original shape `(k, n)` in Marlin format; see `Layer.pack()`
    @C: `torch.half` out matrix of shape `(m, n)` in standard row-major layout
    @s: `torch.half` scales of shape `(m / groupsize, n)`
    @workspace: `torch.int` tensor with at least `n / 128` entries that are all zero
    @thread_k: `k` size of a thread_tile in `B` (can usually be left as auto -1)
    @thread_n: `n` size of a thread_tile in `B` (can usually be left as auto -1)
    @sms: number of SMs to use for the kernel (can usually be left as auto -1)
    """
    marlin_cuda.mul(A, B, C, s, workspace, thread_k, thread_n, sms)


# Precompute permutations for Marlin weight and scale shuffling 

def _get_perms():
    perm = []
    for i in range(32):
        perm1 = []
        col = i // 4
        for block in [0, 1]:
            for row in [
                2 * (i % 4),
                2 * (i % 4) + 1,
                2 * (i % 4 + 4),
                2 * (i % 4 + 4) + 1
            ]:
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm.extend([p + 256 * j for p in perm1])

    perm = np.array(perm)
    interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    perm = perm.reshape((-1, 8))[:, interleave].ravel()
    perm = torch.from_numpy(perm)
    scale_perm = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return perm, scale_perm, scale_perm_single

_perm, _scale_perm, _scale_perm_single = _get_perms()


class WQLinear_Marlin(nn.Module):
    """PyTorch compatible Marlin layer; 4-bit (symmetric grouped) linear layer without bias."""

    def __init__(self, infeatures, outfeatures, groupsize=-1):
        """Create an empty Marlin layer.
        @infeatures: number of input features (must be divisible by 128)
        @outfeatures: number of output features (must be divisible by 256)
        @groupsize: quantization groupsize (must be -1 or 128)
        """
        super().__init__()
        if groupsize not in [-1, 128]:
            raise ValueError('Only groupsize -1 and 128 are supported.')
        if infeatures % 128 != 0 or outfeatures != 256 == 0:
            raise ValueError('`infeatures` must be divisible by 128 and `outfeatures` by 256.')
        if groupsize == -1:
            groupsize = infeatures
        if infeatures % groupsize != 0:
            raise ValueError('`infeatures` must be divisible by `groupsize`.')
        self.k = infeatures
        self.n = outfeatures
        self.groupsize = groupsize
        self.register_buffer('B', torch.empty((self.k // 16, self.n * 16 // 8), dtype=torch.int))
        self.register_buffer('s', torch.empty((self.k // groupsize, self.n), dtype=torch.half))
        # 128 is currently the minimum `tile_n`, hence it gives the maximum workspace size
        self.register_buffer('workspace', torch.zeros(self.n // 128, dtype=torch.int), persistent=False)

    def forward(self, A):
        C = torch.empty(A.shape[:-1] + (self.s.shape[1],), dtype=A.dtype, device=A.device)
        mul(A.view((-1, A.shape[-1])), self.B, C.view((-1, C.shape[-1])), self.s, self.workspace)
        return C

    def pack(self, linear, scales):
        """Pack a fake-quantized linear layer into this actual Marlin representation.
        @linear: fake-quantized `torch.nn.Linear` layer to convert (must be of type `torch.half`)
        @scales: corresponding quantization scales of shape `(infeatures, groups)`
        """ 
        if linear.weight.dtype != torch.half:
            raise ValueError('Only `torch.half` weights are supported.')
        tile = 16
        maxq = 2 ** 4 - 1
        s = scales.t()
        w = linear.weight.data.t()
        if self.groupsize != self.k:
            w = w.reshape((-1, self.groupsize, self.n))
            w = w.permute(1, 0, 2)
            w = w.reshape((self.groupsize, -1))
            s = s.reshape((1, -1))
        w = torch.round(w / s).int()
        w += (maxq + 1) // 2
        w = torch.clamp(w, 0, maxq)
        if self.groupsize != self.k:
            w = w.reshape((self.groupsize, -1, self.n))
            w = w.permute(1, 0, 2)
            w = w.reshape((self.k, self.n)).contiguous()
            s = s.reshape((-1, len(_scale_perm)))[:, _scale_perm]
        else:
            s = s.reshape((-1, len(_scale_perm_single)))[:, _scale_perm_single]
        s = s.reshape((-1, self.n)).contiguous()
        w = w.reshape((self.k // tile, tile, self.n // tile, tile))
        w = w.permute((0, 2, 1, 3))
        w = w.reshape((self.k // tile, self.n * tile))
        res = w
        res = res.reshape((-1, _perm.numel()))[:, _perm].reshape(res.shape)
        q = np.zeros((res.shape[0], res.shape[1] // 8), dtype=np.uint32)
        res = res.cpu().numpy().astype(np.uint32)
        for i in range(8):
            q |= res[:, i::8] << 4 * i
        q = torch.from_numpy(q.astype(np.int32)).to(w.device)
        self.B[:, :] = q.to(self.B.device)
        self.s[:, :] = s.to(self.s.device)
