import torch
# from .triton_on2 import flash_rnn_on2
from einops import rearrange
import triton 
import triton.language as tl
from cuda import cuda_compute_intra
from cuda import cuda_compute_intra_chunk64x_d64x
import torch.nn.functional as F


@triton.jit
def _fwd_recurrence(
    S, G_K, G_V, K, V,  
    CHUNK_SIZE: tl.constexpr,
    D_MODEL_K: tl.constexpr, D_MODEL_V: tl.constexpr,
    BLOCK_MODEL: tl.constexpr
  ):
    offset_bh = tl.program_id(0)
    offset_k = tl.program_id(1)
    offset_v = tl.program_id(2)    

    S = S + offset_bh * CHUNK_SIZE * D_MODEL_K * D_MODEL_V + offset_k * D_MODEL_V * BLOCK_MODEL + tl.arange(0, BLOCK_MODEL)[:, None] * D_MODEL_V + offset_v * BLOCK_MODEL + tl.arange(0, BLOCK_MODEL)[None, :]

    G_K = G_K + offset_bh * CHUNK_SIZE * D_MODEL_K  + tl.arange(0, BLOCK_MODEL) + offset_k * BLOCK_MODEL      

    K = K + offset_bh * CHUNK_SIZE * D_MODEL_K + tl.arange(0, BLOCK_MODEL) + offset_k * BLOCK_MODEL      

    G_V = G_V + offset_bh * CHUNK_SIZE * D_MODEL_V + tl.arange(0, BLOCK_MODEL) + offset_v * BLOCK_MODEL  

    V = V + offset_bh * CHUNK_SIZE * D_MODEL_V + tl.arange(0, BLOCK_MODEL) + offset_v * BLOCK_MODEL 

    acc = tl.zeros([BLOCK_MODEL, BLOCK_MODEL], dtype=tl.float32)

    for i in range(CHUNK_SIZE):
        g_k = tl.load(G_K)
        g_v = tl.load(G_V)
        k = tl.load(K)
        v = tl.load(V)

        acc = acc * g_k[:, None] * g_v[None, :] + k[:, None] * v[None, :]

        tl.store(S, acc.to(S.dtype.element_ty))
        
        G_K += D_MODEL_K
        K += D_MODEL_K
        G_V += D_MODEL_V
        V += D_MODEL_V
        S += D_MODEL_K * D_MODEL_V
    

@triton.jit
def _bwd_recurrence(
    S, DS, G_K, G_V, Q, DO,   
    CHUNK_SIZE,
    D_MODEL_K: tl.constexpr, D_MODEL_V: tl.constexpr,
    BLOCK_MODEL: tl.constexpr
  ):
    offset_bh = tl.program_id(0)
    offset_k = tl.program_id(1)
    offset_v = tl.program_id(2)    

    S = S + (offset_bh * CHUNK_SIZE + CHUNK_SIZE - 2) * D_MODEL_K * D_MODEL_V +  offset_k * D_MODEL_V * BLOCK_MODEL + tl.arange(0, BLOCK_MODEL)[:, None] * D_MODEL_V + offset_v * BLOCK_MODEL + tl.arange(0, BLOCK_MODEL)[None, :]


    DS = DS + (offset_bh * CHUNK_SIZE + CHUNK_SIZE - 1) * D_MODEL_K * D_MODEL_V + offset_k * D_MODEL_V * BLOCK_MODEL + tl.arange(0, BLOCK_MODEL)[:, None] * D_MODEL_V + offset_v * BLOCK_MODEL + tl.arange(0, BLOCK_MODEL)[None, :]
    
    G_K = G_K + (offset_bh * CHUNK_SIZE + CHUNK_SIZE - 1) * D_MODEL_K  + tl.arange(0, BLOCK_MODEL) + offset_k * BLOCK_MODEL      

    Q = Q + (offset_bh * CHUNK_SIZE + CHUNK_SIZE - 1) * D_MODEL_K + tl.arange(0, BLOCK_MODEL) + offset_k * BLOCK_MODEL      

    DO = DO + (offset_bh * CHUNK_SIZE + CHUNK_SIZE - 1) * D_MODEL_V + tl.arange(0, BLOCK_MODEL) + offset_v * BLOCK_MODEL

    G_V = G_V + (offset_bh * CHUNK_SIZE + CHUNK_SIZE - 1) * D_MODEL_V + tl.arange(0, BLOCK_MODEL) + offset_v * BLOCK_MODEL  

    acc = tl.zeros([BLOCK_MODEL, BLOCK_MODEL], dtype=tl.float32)


    for i in range(CHUNK_SIZE-1, -1, -1):

        q = tl.load(Q)

        do = tl.load(DO)

        g_k = tl.load(G_K)

        g_v = tl.load(G_V)

        acc += q[:, None] * do[None, :]

        if i > 0:
            s = tl.load(S)          
            tl.store(S + D_MODEL_K * D_MODEL_V, (s*acc).to(S.dtype.element_ty))
        else:
            tl.store(S + D_MODEL_K * D_MODEL_V, tl.zeros_like(acc).to(S.dtype.element_ty))

        tl.store(DS, acc.to(DS.dtype.element_ty))                
        acc = acc * g_k[:, None] * g_v[None, :] 
        G_K -= D_MODEL_K
        Q -= D_MODEL_K
        G_V -= D_MODEL_V 
        DO -= D_MODEL_V
        S -= D_MODEL_K * D_MODEL_V
        DS -= D_MODEL_K * D_MODEL_V


class IntraChunkFusedLoop(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, gk, gv):
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        gk = gk.contiguous()
        gv = gv.contiguous()

        B, H, N, CHUNK_SIZE, D = q.shape

        D_k = k.shape[-1]
        D_v = v.shape[-1]
        
        # (B, H, L, D_K, D_V)
        S = torch.empty(q.shape[0], q.shape[1], q.shape[2], q.shape[3], k.shape[-1], v.shape[-1], device=q.device, dtype=q.dtype).contiguous()

        BLOCK_MODEL = 32
    
        assert D_k % BLOCK_MODEL == 0
        assert D_v % BLOCK_MODEL == 0

        grid = ( B * H * N,  D_k//BLOCK_MODEL, D_v//BLOCK_MODEL)
        ctx.grid = grid 
        ctx.BLOCK_MODEL = BLOCK_MODEL


        _fwd_recurrence[grid](
            S, gk, gv, k, v,            
            CHUNK_SIZE=CHUNK_SIZE, 
            D_MODEL_K=D_k, D_MODEL_V=D_v,
            BLOCK_MODEL=BLOCK_MODEL
        )
        
        ctx.save_for_backward(q, k, v, gk, gv, S)  
        ctx.grid = grid 
        ctx.BLOCK_MODEL = BLOCK_MODEL
        ctx.CHUNK_SIZE=CHUNK_SIZE
        ctx.D_MODEL_K = D_k
        ctx.D_MODEL_V = D_v                
        return   torch.einsum('b h n c k v, b h n c k -> b h n c v', S, q)


    @staticmethod
    def backward(ctx, DO):
        DO = DO.contiguous()

        q, k, v, gk, gv, S = ctx.saved_tensors 
        DS = torch.empty_like(S).contiguous()
        
        S = S.contiguous()

        D_k = k.shape[-1]
        D_v = v.shape[-1]


        dq = torch.einsum('b h n c k v, b h n c v -> b h n c k', S, DO)
        B, H, N, CHUNK_SIZE, D = q.shape
        BLOCK_MODEL = 32


        BLOCK_MODEL = ctx.BLOCK_MODEL 

        _bwd_recurrence[ctx.grid](
            S.contiguous(), DS.contiguous(), gk.contiguous(), gv.contiguous(), q.contiguous(), DO.contiguous(),                      
            CHUNK_SIZE=CHUNK_SIZE, 
            D_MODEL_K=D_k, D_MODEL_V=D_v,
            BLOCK_MODEL=BLOCK_MODEL
        )

        dk = torch.einsum('b h n c k v, b h n c v -> b h n c k', DS, v)
        dv = torch.einsum('b h n c k v, b h n c k -> b h n c v', DS, k)

        dgk = torch.einsum('b h n c k v, b h n c v -> b h n c k', S, gv)
        dgv = torch.einsum('b h n c k v, b h n c k -> b h n c v', S, gk)

        return dq, dk, dv, dgk, dgv

def naive_recurrence(q, k, v, gk, gv):
    B, H, N, C, D_K = k.shape
    _, _, _, _, D_V = v.shape

    S = torch.zeros(B, H, N, C, D_K, D_V, device=q.device)
    H = torch.zeros(B, H, D_K, D_V, device=q.device)
    
    for n in range(N):
        for i in range(C):
            if i == 0:
                H = k[:, :, n, i, :, None] * v[:, :, n, i, None, :]            
            else:
                H = k[:, :, n, i, :, None] * v[:, :, n, i, None, :] + H * gk[:, :, n, i, :, None] * gv[:, :, n, i, None, :]
            

            S[:, :, n, i] = H 

    return  torch.einsum('b h  n c k v, b h n c k -> b h n c v', S, q)



if __name__ == "__main__":
    B, H, N, C, D_K, D_V = 2, 2, 16, 128, 64, 64
    # print("?")


    require_grad = True
    q = torch.randn(B, H, N,C, D_K, device='cuda').requires_grad_(require_grad)
    k = torch.randn(B, H, N,C, D_K, device='cuda').requires_grad_(require_grad)
    v = torch.randn(B, H, N,C, D_V, device='cuda').requires_grad_(require_grad)
    gk = torch.randn(B, H, N,C, D_K, device='cuda').sigmoid().requires_grad_(require_grad)
    gv = torch.randn(B, H, N,C, D_V, device='cuda').sigmoid().requires_grad_(require_grad)

    target = [q, k, v, gk, gv]
    p1 = []
    p2 = []
    
    o = IntraChunkFusedLoop.apply(q, k, v, gk, gv)
    o.sum().backward()

    for p in target:
        p1.append(p.grad.clone())
        p.grad.zero_()
       
    o2 = naive_recurrence(q, k, v, gk, gv)
    o2.sum().backward()

    assert torch.isclose(o, o2, atol=1e-3).all()
    for p in target:
        p2.append(p.grad.clone())
        p.grad.zero_()


    for (g1, g2) in zip(p1, p2):
        print((g1-g2).abs().max())
    









