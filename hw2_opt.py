import torch
import triton
import triton.language as tl
import triton.testing

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def _ln_fwd_kernel(X, W, B, Out, Mean, Rstd, Xhat,
                   stride_x, stride_w, stride_b, stride_out,
                   N, eps, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    row_start = pid * stride_x
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(X + row_start + offsets, mask=mask, other=0.0)
    w = tl.load(W + offsets, mask=mask, other=0.0)
    b = tl.load(B + offsets, mask=mask, other=0.0)

    sum_x = tl.sum(tl.where(mask, x, 0.0))
    mean = sum_x / N
    var = tl.sum(tl.where(mask, (x - mean) * (x - mean), 0.0)) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    x_hat = (x - mean) * rstd
    out = x_hat * w + b

    tl.store(Out + row_start + offsets, out, mask=mask)
    tl.store(Mean + pid, mean)
    tl.store(Rstd + pid, rstd)
    tl.store(Xhat + row_start + offsets, x_hat, mask=mask)

@triton.jit
def _ln_bwd_kernel(Dy, W, Xhat, Rstd, DX, DW, DB,
                   stride_dy, stride_dx, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    row_start = pid * stride_dy
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    dy = tl.load(Dy + row_start + offsets, mask=mask, other=0.0)
    w = tl.load(W + offsets, mask=mask, other=0.0)
    x_hat = tl.load(Xhat + row_start + offsets, mask=mask, other=0.0)
    rstd = tl.load(Rstd + pid)

    dy_w = dy * w

    mean_dy_w = tl.sum(tl.where(mask, dy_w, 0.0)) / N
    mean_dy_w_xhat = tl.sum(tl.where(mask, dy_w * x_hat, 0.0)) / N

    dx = rstd * (dy_w - mean_dy_w - x_hat * mean_dy_w_xhat)

    tl.store(DX + row_start + offsets, dx, mask=mask)
    tl.atomic_add(DW + offsets, dy * x_hat, mask=mask)  # FIX: было dy_w * x_hat
    tl.atomic_add(DB + offsets, dy, mask=mask)

class LayerNormTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps=1e-5):
        x = x.contiguous()
        weight = weight.contiguous()
        bias = bias.contiguous()
        M, N = x.shape

        mean = torch.empty(M, device=x.device, dtype=x.dtype)
        rstd = torch.empty(M, device=x.device, dtype=x.dtype)
        x_hat = torch.empty_like(x)
        out = torch.empty_like(x)

        grid = (M,)
        _ln_fwd_kernel[grid](
            x, weight, bias, out, mean, rstd, x_hat,
            x.stride(0), weight.stride(0), bias.stride(0), out.stride(0),
            N, eps
        )

        ctx.save_for_backward(x_hat, weight, rstd)
        ctx.N = N
        return out

    @staticmethod
    def backward(ctx, dy):
        x_hat, weight, rstd = ctx.saved_tensors
        dy = dy.contiguous()
        M, N = dy.shape

        dx = torch.empty_like(dy)
        dw = torch.zeros_like(weight)
        db = torch.zeros_like(weight)

        grid = (M,)
        _ln_bwd_kernel[grid](
            dy, weight, x_hat, rstd,
            dx, dw, db,
            dy.stride(0), dx.stride(0),
            N, triton.next_power_of_2(N)
        )

        return dx, dw, db, None

def layernorm_forward_torch(x, weight, bias, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    rstd = 1.0 / torch.sqrt(var + eps)
    x_hat = (x - mean) * rstd
    return x_hat * weight + bias

def test_and_benchmark():
    torch.manual_seed(42)
    M, N = 1024, 1024
    dtype = torch.float32
    device = 'cuda'

    x = torch.randn(M, N, device=device, dtype=dtype, requires_grad=True)
    w = torch.randn(N, device=device, dtype=dtype, requires_grad=True)
    b = torch.randn(N, device=device, dtype=dtype, requires_grad=True)

    x_triton = x.clone().detach().requires_grad_(True)
    w_triton = w.clone().detach().requires_grad_(True)
    b_triton = b.clone().detach().requires_grad_(True)

    out_torch = layernorm_forward_torch(x, w, b)
    out_triton = LayerNormTriton.apply(x_triton, w_triton, b_triton)

     
    torch.testing.assert_close(out_triton, out_torch, atol=1e-4, rtol=1e-4)
     

    loss_torch = out_torch.sum()
    loss_torch.backward()

    loss_triton = out_triton.sum()
    loss_triton.backward()

     
    torch.testing.assert_close(x_triton.grad, x.grad, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(w_triton.grad, w.grad, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(b_triton.grad, b.grad, atol=1e-4, rtol=1e-4)
    

     
    fn_torch = lambda: layernorm_forward_torch(x, w, b)
    fn_triton = lambda: LayerNormTriton.apply(x_triton, w_triton, b_triton)

    time_torch = triton.testing.do_bench(fn_torch, warmup=10, rep=100)
    time_triton = triton.testing.do_bench(fn_triton, warmup=10, rep=100)

    print(f"PyTorch: {time_torch:.3f} ms")
    print(f"Triton : {time_triton:.3f} ms")
    print(f"Speedup: {time_torch/time_triton:.2f}x")

if __name__ == '__main__':
    test_and_benchmark()
