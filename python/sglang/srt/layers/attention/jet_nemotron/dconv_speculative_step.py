import torch
import triton
import triton.language as tl


@triton.jit
def _causal_conv_speculative_step_kernel(
    X_ptr,  # Pointer to current input x [B, T, D]
    Cache_ptr,  # Pointer to cache [N, D, W]
    Kernels_ptr,  # Pointer to generated kernels [B, T, D, W]
    Out_ptr,  # Pointer to output tensor [B, T, D]
    Cache_idx_ptr,  # Pointer to cache indices [B]
    Intermediate_conv_window_ptr,  # Pointer to intermediate conv window [N, W-1+T, D]
    B,
    D,
    X_stride_b,
    X_stride_t,
    X_stride_d,
    Cache_stride_b,
    Cache_stride_d,
    Cache_stride_w,
    Kernels_stride_b,
    Kernels_stride_t,
    Kernels_stride_d,
    Kernels_stride_w,
    Out_stride_b,
    Out_stride_t,
    Out_stride_d,
    Cache_idx_stride_b,
    Intermediate_conv_window_stride_b,
    Intermediate_conv_window_stride_t,
    Intermediate_conv_window_stride_d,
    W: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_time = tl.program_id(1)
    pid_d_block = tl.program_id(2)
    cache_idx = tl.load(Cache_idx_ptr + pid_batch * Cache_idx_stride_b)
    offs_d = pid_d_block * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
    d_mask = offs_d < D
    accumulator = tl.zeros((BLOCK_SIZE_D,), dtype=tl.float32)
    offs_w = tl.arange(0, W)
    k_ptrs = Kernels_ptr + (
        pid_batch * Kernels_stride_b
        + pid_time * Kernels_stride_t
        + offs_d[:, None] * Kernels_stride_d
        + offs_w[None, :] * Kernels_stride_w
    )
    k_vals = tl.load(k_ptrs, mask=d_mask[:, None], other=0.0)
    x_abs_indices = pid_time + offs_w - W + 1
    x_ptrs = X_ptr + (
        pid_batch * X_stride_b
        + x_abs_indices[None, :] * X_stride_t
        + offs_d[:, None] * X_stride_d
    )
    x_final_load_mask = d_mask[:, None] & (x_abs_indices >= 0)[None, :]
    x_input_vals = tl.load(x_ptrs, mask=x_final_load_mask, other=0.0)
    cache_ptrs = Cache_ptr + (
        cache_idx * Cache_stride_b
        + (x_abs_indices + W)[None, :] * Cache_stride_w
        + offs_d[:, None] * Cache_stride_d
    )
    cache_final_load_mask = d_mask[:, None] & (x_abs_indices < 0)[None, :]
    vals_from_cache = tl.load(cache_ptrs, mask=cache_final_load_mask, other=0.0)
    x_vals = x_input_vals + vals_from_cache
    product = k_vals * x_vals
    accumulator += tl.sum(product, axis=1)
    out_ptrs = Out_ptr + (
        pid_batch * Out_stride_b + pid_time * Out_stride_t + offs_d * Out_stride_d
    )
    tl.store(out_ptrs, accumulator, mask=d_mask)

    intermediate_conv_window_ptrs = Intermediate_conv_window_ptr + (
        cache_idx * Intermediate_conv_window_stride_b
        + (W - 2 + pid_time) * Intermediate_conv_window_stride_t
        + offs_d * Intermediate_conv_window_stride_d
    )
    offs_w = tl.arange(0, W)
    last_col_mask = offs_w == W - 1
    x_vals_last_col = tl.sum(x_vals * last_col_mask[None, :], axis=1)
    tl.store(intermediate_conv_window_ptrs, x_vals_last_col, mask=d_mask)


def causal_conv_step_triton_speculative(
    x: torch.Tensor,  # Input tensor [B, T, D]
    cache: torch.Tensor,  # Cache tensor [N, D, W]
    kernels: torch.Tensor,  # Kernels tensor [B, T, D, W]
    cache_indices: torch.Tensor,  # Cache indices tensor [B]
    intermediate_conv_window: torch.Tensor,  # Intermediate conv window tensor [N, W-2+T, D], updated in-place
) -> torch.Tensor:  # Returns output tensor [B, D] (before activation)
    B, T, D = x.shape
    W = cache.shape[2]
    assert 1 < W <= 4, f"Kernel W={W}, this optimized version assumes 1 < W <= 4"
    assert (
        x.is_cuda and cache.is_cuda and kernels.is_cuda
    ), "Inputs must be CUDA tensors"

    out = torch.empty_like(x)

    grid = lambda meta: (B, T, triton.cdiv(D, meta["BLOCK_SIZE_D"]))
    BLOCK_SIZE_D = 64

    _causal_conv_speculative_step_kernel[grid](
        x,
        cache,
        kernels,
        out,
        cache_indices,
        intermediate_conv_window,
        B,
        D,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        cache.stride(0),
        cache.stride(1),
        cache.stride(2),
        kernels.stride(0),
        kernels.stride(1),
        kernels.stride(2),
        kernels.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        cache_indices.stride(0),
        intermediate_conv_window.stride(0),
        intermediate_conv_window.stride(1),
        intermediate_conv_window.stride(2),
        W=W,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
    )

    return out
