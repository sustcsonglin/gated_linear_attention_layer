import torch
# from .triton_on2 import flash_rnn_on2
from einops import rearrange
import triton 
import triton.language as tl
from cuda import cuda_compute_intra
from cuda import cuda_compute_intra_chunk64x_d64x
import torch.nn.functional as F
from pytorch_chunk_onc import Chunk_memory_update


@triton.jit
def _fwd_recurrence(
    S, G_K, G_V, Q, K, V, O,
    
    CHUNK_SIZE: tl.constexpr, NUM_CHUNK: tl.constexpr,
    L: tl.constexpr,
    D_MODEL_K: tl.constexpr, D_MODEL_V: tl.constexpr,
    BLOCK_MODEL_K: tl.constexpr, 
    BLOCK_MODEL_V: tl.constexpr,
    SAVE_S: tl.constexpr
  ):
    offset_bh = tl.program_id(0)
    offset_k = tl.program_id(1)
    offset_v = tl.program_id(2)    

    S = S + offset_bh * NUM_CHUNK * D_MODEL_K * D_MODEL_V + offset_k * D_MODEL_V * BLOCK_MODEL_K + tl.arange(0, BLOCK_MODEL_K)[:, None] * D_MODEL_V + offset_v * BLOCK_MODEL_V + tl.arange(0, BLOCK_MODEL_V)[None, :] + D_MODEL_V * D_MODEL_K

    G_K = G_K + offset_bh * NUM_CHUNK * D_MODEL_K + tl.arange(0, BLOCK_MODEL_K) + offset_k *  BLOCK_MODEL_K      

    K = K + offset_bh * L * D_MODEL_K + tl.arange(0, CHUNK_SIZE)[None, :] * D_MODEL_K + offset_k * BLOCK_MODEL_K + tl.arange(0, BLOCK_MODEL_K)[:, None]

    V = V + offset_bh * L * D_MODEL_V + tl.arange(0, CHUNK_SIZE)[:, None] * D_MODEL_V + offset_v * BLOCK_MODEL_V + tl.arange(0, BLOCK_MODEL_V)[None, :]

    Q = Q + offset_bh * L * D_MODEL_K + tl.arange(0, CHUNK_SIZE)[:, None] * D_MODEL_K + offset_k * BLOCK_MODEL_K + tl.arange(0, BLOCK_MODEL_K)[None, :] + D_MODEL_K * CHUNK_SIZE

    O = O + offset_bh * L * D_MODEL_V * (D_MODEL_K // BLOCK_MODEL_K) + offset_k * L * D_MODEL_V +  tl.arange(0, CHUNK_SIZE)[:, None] * D_MODEL_V + offset_v * BLOCK_MODEL_V + tl.arange(0, BLOCK_MODEL_V)[None, :] + D_MODEL_V * CHUNK_SIZE

    G_V = G_V + offset_bh * NUM_CHUNK * D_MODEL_V + tl.arange(0, BLOCK_MODEL_V) + offset_v * BLOCK_MODEL_V  

    # 64 * 64
    acc = tl.zeros([BLOCK_MODEL_K, BLOCK_MODEL_V], dtype=tl.float32)

    for i in range(NUM_CHUNK-1):
        g_k = tl.load(G_K)
        g_v = tl.load(G_V)
        k = tl.load(K) * tl.load(reduce_chunk_key)
        v = tl.load(V) * tl.load(reduce_chunk_value)        
        
        acc = acc * g_k[:, None] * g_v[None, :] + tl.dot(k, v, allow_tf32=False)

        if SAVE_S:
            tl.store(S, acc.to(S.dtype.element_ty))
        
        q = tl.load(Q) 
        
        o = tl.dot(q, acc.to(q.dtype), allow_tf32=False)        
        tl.store(O, o.to(O.dtype.element_ty))

        K += D_MODEL_K * CHUNK_SIZE     
        Q += D_MODEL_K * CHUNK_SIZE
        V += D_MODEL_V * CHUNK_SIZE
        O += D_MODEL_V * CHUNK_SIZE
        
        G_V += D_MODEL_V
        G_K += D_MODEL_K
        S += D_MODEL_K * D_MODEL_V
    

@triton.jit
def _bwd_recurrence(
    S, DS, G_K, G_V, Q,  
    DO, 
    DG_K, DG_V,  

    CHUNK_SIZE: tl.constexpr, NUM_CHUNK: tl.constexpr,
    L: tl.constexpr,
    D_MODEL_K: tl.constexpr, D_MODEL_V: tl.constexpr,
    BLOCK_MODEL_K: tl.constexpr,
    BLOCK_MODEL_V: tl.constexpr,
  ):
    
    offset_bh = tl.program_id(0)
    offset_k = tl.program_id(1)
    offset_v = tl.program_id(2)    
    
    NUM_SPLIT_K = (D_MODEL_K // BLOCK_MODEL_K)
    NUM_SPLIT_V = (D_MODEL_V // BLOCK_MODEL_V)

    S = S + offset_bh * NUM_CHUNK * D_MODEL_K * D_MODEL_V + offset_k * D_MODEL_V * BLOCK_MODEL_K + tl.arange(0, BLOCK_MODEL_K)[:, None] * D_MODEL_V + offset_v * BLOCK_MODEL_V + tl.arange(0, BLOCK_MODEL_V)[None, :] + (NUM_CHUNK - 2) * D_MODEL_K * D_MODEL_V

    DS = DS + offset_bh * NUM_CHUNK * D_MODEL_K * D_MODEL_V + offset_k * D_MODEL_V * BLOCK_MODEL_K + tl.arange(0, BLOCK_MODEL_K)[:, None] * D_MODEL_V + offset_v * BLOCK_MODEL_V + tl.arange(0, BLOCK_MODEL_V)[None, :] + (NUM_CHUNK - 1) * D_MODEL_K * D_MODEL_V

    G_K = G_K + offset_bh * NUM_CHUNK * D_MODEL_K + tl.arange(0, BLOCK_MODEL_K) + offset_k * BLOCK_MODEL_K + (NUM_CHUNK - 2) * D_MODEL_K

    DG_K = DG_K + offset_bh * NUM_CHUNK * D_MODEL_K * NUM_SPLIT_V + offset_v * NUM_CHUNK * D_MODEL_K  + tl.arange(0, BLOCK_MODEL_K) + offset_k * BLOCK_MODEL_K + (NUM_CHUNK - 2) * D_MODEL_K 

    DO = DO + offset_bh * L * D_MODEL_V + tl.arange(0, CHUNK_SIZE)[:, None] * D_MODEL_V + offset_v * BLOCK_MODEL_V + tl.arange(0, BLOCK_MODEL_V)[None, :] + (NUM_CHUNK - 1) * D_MODEL_V * CHUNK_SIZE

    Q = Q + offset_bh * L * D_MODEL_K + tl.arange(0, CHUNK_SIZE)[None, :] * D_MODEL_K + offset_k * BLOCK_MODEL_K + tl.arange(0, BLOCK_MODEL_K)[:, None]  + (NUM_CHUNK - 1) * D_MODEL_K * CHUNK_SIZE

    G_V = G_V + offset_bh * NUM_CHUNK * D_MODEL_V + tl.arange(0, BLOCK_MODEL_V) + offset_v * BLOCK_MODEL_V + (NUM_CHUNK - 2) * D_MODEL_V

    DG_V = DG_V + offset_bh * NUM_CHUNK * D_MODEL_V * NUM_SPLIT_K + offset_k * NUM_CHUNK * D_MODEL_V + tl.arange(0, BLOCK_MODEL_V) + offset_v * BLOCK_MODEL_V + (NUM_CHUNK - 2) * D_MODEL_V

    # 64 * 64
    acc = tl.zeros([BLOCK_MODEL_K, BLOCK_MODEL_V], dtype=tl.float32)

    # s = tl.load(S)

    for i in range(NUM_CHUNK-1):
        q = tl.load(Q) 
        do = tl.load(DO)

        g_k = tl.load(G_K)
        g_v = tl.load(G_V)
        s = tl.load(S)

        # k = tl.load(K)
        # v = tl.load(V)

        # dq = tl.dot(s, tl.trans(do), allow_tf32=False)
        # tl.store(DQ, dq.to(DQ.dtype.element_ty))


        acc += tl.dot(q, do, allow_tf32=False)        

        tl.store(DS, acc)

        ds = s * acc    
        dgv = tl.sum(ds * g_k[:, None], axis=0)
        dgk = tl.sum(ds * g_v[None, :], axis=1)
        
        tl.store(DG_V, dgv.to(DG_V.dtype.element_ty))
        tl.store(DG_K, dgk.to(DG_K.dtype.element_ty))

        acc = acc * g_k[:, None] * g_v[None, :]

        Q -= D_MODEL_K * CHUNK_SIZE 
        DO -= D_MODEL_V * CHUNK_SIZE

        G_V -= D_MODEL_V
        G_K -= D_MODEL_K
        DG_K -= D_MODEL_K
        DG_V -= D_MODEL_V        
    
        S -= D_MODEL_K * D_MODEL_V
        DS -= D_MODEL_K * D_MODEL_V



class InterChunkFused(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, gk, gv):
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        gk = gk.contiguous()
        gv = gv.contiguous()

        B, H, NUM_CHUNK, CHUNK_SIZE, D = q.shape

        D_k = k.shape[-1]
        D_v = v.shape[-1]
        
        # (B, H, L, D_K, D_V)
        S = torch.empty(q.shape[0], q.shape[1], q.shape[2], k.shape[-1], v.shape[-1], device=q.device, dtype=q.dtype, memory_format=torch.contiguous_format)

        # o = torch.empty_like(v).contiguous()

        BLOCK_MODEL_K = 128
        BLOCK_MODEL_V = 16
    
        #split k
        o = torch.empty(B, H, D_k // BLOCK_MODEL_K, NUM_CHUNK, CHUNK_SIZE, D_v, device=q.device, dtype=q.dtype, memory_format=torch.contiguous_format)
    
        assert D_k % BLOCK_MODEL_K == 0
        assert D_v % BLOCK_MODEL_V == 0

        grid = (B * H,  D_k//BLOCK_MODEL_K, D_v//BLOCK_MODEL_V)
        ctx.grid = grid 

        # ctx.BLOCK_MODEL = BLOCK_MODEL

        _fwd_recurrence[grid](
            S, gk, gv, q, k, v, o,
            CHUNK_SIZE=CHUNK_SIZE, NUM_CHUNK = NUM_CHUNK, L = CHUNK_SIZE * NUM_CHUNK,
            D_MODEL_K=D_k, D_MODEL_V=D_v,
            BLOCK_MODEL_K=BLOCK_MODEL_K, BLOCK_MODEL_V=BLOCK_MODEL_V, SAVE_S=True
        )

        ctx.save_for_backward(q, k, v, gk, gv, S)  
        ctx.grid = grid 
        # ctx.BLOCK_MODEL = BLOCK_MODEL
        ctx.CHUNK_SIZE=CHUNK_SIZE
        ctx.D_MODEL_K = D_k
        ctx.D_MODEL_V = D_v                
        o[:, :, :, 0] = 0
        return o.sum(2)



    @staticmethod
    def backward(ctx, DO):
        q, k, v, gk, gv, S = ctx.saved_tensors

        DO = DO.contiguous()
        B, H, NUM_CHUNK, CHUNK_SIZE, D_v = DO.shape
        D_k = k.shape[-1]
        D_v = v.shape[-1]
        BLOCK_MODEL_K = 128
        BLOCK_MODEL_V = 16

        dgk = torch.empty(B, H, D_v//BLOCK_MODEL_V, NUM_CHUNK, D_k, device=q.device, dtype=q.dtype, memory_format=torch.contiguous_format)
        dgv = torch.empty(B, H, D_k//BLOCK_MODEL_K, NUM_CHUNK, D_v, device=q.device, dtype=q.dtype, memory_format=torch.contiguous_format)

        grid = (B * H,  D_k//BLOCK_MODEL_K, D_v//BLOCK_MODEL_V)
        
        DS = torch.empty_like(S)

        _bwd_recurrence[grid](
                S, DS, gk, gv, q,  
                DO, 
                dgk, dgv, 
                CHUNK_SIZE=CHUNK_SIZE, NUM_CHUNK = NUM_CHUNK, L = CHUNK_SIZE * NUM_CHUNK,
                D_MODEL_K=D_k, D_MODEL_V=D_v,
                BLOCK_MODEL_K=BLOCK_MODEL_K,
                BLOCK_MODEL_V=BLOCK_MODEL_V,
        )


        return torch.einsum('b h n k v, b h n c v -> b h n c k', S, DO), torch.einsum('b h n k v, b h n c v -> b h n c k', DS, v), torch.einsum('b h n k v, b h n c k -> b h n c v', DS, k), dgk.sum(2), dgv.sum(2)


        

def naive(q, k, v, gk, gv):
    to_add = k.transpose(-1, -2) @ v
    S = Chunk_memory_update.apply(gk, gv, to_add, True)    
    return torch.einsum('b h n c k, b h n k v -> b h n c v', q, S)



if __name__ == "__main__":
    B, H, N, C, D_K, D_V = 4, 4, 32, 64, 128, 256
    # print("?")





    require_grad = True
    q = torch.randn(B, H, N, C, D_K, device='cuda').requires_grad_(require_grad)
    k = torch.randn(B, H, N, C, D_K, device='cuda').requires_grad_(require_grad)
    v = torch.randn(B, H, N, C, D_V, device='cuda').requires_grad_(require_grad)
    gk = torch.randn(B, H, N,  D_K, device='cuda').sigmoid().requires_grad_(require_grad)
    gv = torch.randn(B, H, N,  D_V, device='cuda').sigmoid().requires_grad_(require_grad)

    target = [q, k, v, gk, gv]
    p1 = []
    p2 = []
    
    o1 = InterChunkFused.apply(q, k, v, gk, gv)    
    o1.sum().backward()

    for p in target:
        p1.append(p.grad.clone())
        p.grad.zero_()
       
    o2 = naive(q, k, v, gk, gv)
    # assert torch.isclose(o1, o2, atol=1e-3).all()

    o2.sum().backward()

    print( (o1 -o2).abs().max())

    for p in target:
        p2.append(p.grad.clone())
        p.grad.zero_()

    for (g1, g2) in zip(p1, p2):
        print((g1-g2).abs().max())
    

    breakpoint()