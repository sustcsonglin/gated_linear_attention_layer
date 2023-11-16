import torch
# from .triton_on2 import flash_rnn_on2
from einops import rearrange
import triton 
import triton.language as tl
# from cuda import cuda_compute_intra
# from cuda import cuda_compute_intra_chunk64x_d64x
import torch.nn.functional as F
# from pytorch_chunk_onc import Chunk_memory_update
# 


@triton.jit
def _fwd_recurrence(
    S, GK, GV, Q, K, V, O, 
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

    S_ptr = S + offset_bh * NUM_CHUNK * D_MODEL_K * D_MODEL_V + offset_k * D_MODEL_V * BLOCK_MODEL_K + tl.arange(0, BLOCK_MODEL_K)[:, None] * D_MODEL_V + offset_v * BLOCK_MODEL_V + tl.arange(0, BLOCK_MODEL_V)[None, :] 

    K_ptr = K + offset_bh * L * D_MODEL_K + tl.arange(0, CHUNK_SIZE)[None, :] * D_MODEL_K + offset_k * BLOCK_MODEL_K + tl.arange(0, BLOCK_MODEL_K)[:, None]

    GK_ptr = GK + offset_bh * L * D_MODEL_K + tl.arange(0, CHUNK_SIZE)[None, :] * D_MODEL_K + offset_k * BLOCK_MODEL_K + tl.arange(0, BLOCK_MODEL_K)[:, None]

    GK_last_ptr = GK + offset_bh * L * D_MODEL_K + (CHUNK_SIZE - 1) * D_MODEL_K + offset_k * BLOCK_MODEL_K + tl.arange(0, BLOCK_MODEL_K) 

    V_ptr = V + offset_bh * L * D_MODEL_V + tl.arange(0, CHUNK_SIZE)[:, None] * D_MODEL_V + offset_v * BLOCK_MODEL_V + tl.arange(0, BLOCK_MODEL_V)[None, :]

    O_ptr = O + offset_bh * L * D_MODEL_V * (D_MODEL_K // BLOCK_MODEL_K) + offset_k * L * D_MODEL_V +  tl.arange(0, CHUNK_SIZE)[:, None] * D_MODEL_V + offset_v * BLOCK_MODEL_V + tl.arange(0, BLOCK_MODEL_V)[None, :] 

    GV_ptr = GV + offset_bh * L * D_MODEL_V + tl.arange(0, CHUNK_SIZE)[:, None] * D_MODEL_V + offset_v * BLOCK_MODEL_V + tl.arange(0, BLOCK_MODEL_V)[None, :]

    GV_last_ptr = GV + offset_bh * L * D_MODEL_V + (CHUNK_SIZE - 1) * D_MODEL_V + offset_v * BLOCK_MODEL_V + tl.arange(0, BLOCK_MODEL_V)

    Q_ptr = Q + offset_bh * L * D_MODEL_K + tl.arange(0, CHUNK_SIZE)[:, None] * D_MODEL_K + offset_k * BLOCK_MODEL_K + tl.arange(0, BLOCK_MODEL_K)[None, :]  

    # # 128 * 128?
    acc = tl.zeros([BLOCK_MODEL_K, BLOCK_MODEL_V], dtype=tl.float32)

    for i in range(NUM_CHUNK):
        ## save output
        gk = tl.load(GK_ptr).to(tl.float32)
        gv = tl.load(GV_ptr).to(tl.float32)

        if i > 0:
            q = tl.load(Q_ptr)
            q = q * tl.exp(tl.trans(gk)).to(q.dtype)               
            output = tl.dot(q, acc.to(q.dtype), allow_tf32=False) * tl.exp(gv)
            tl.store(O_ptr, output.to(O_ptr.dtype.element_ty))

        if i < NUM_CHUNK:
            gk_last = tl.load(GK_last_ptr).to(tl.float32)
            gv_last = tl.load(GV_last_ptr).to(tl.float32)

            k = tl.load(K_ptr)
            k = k * tl.exp(gk_last[:, None] - gk).to(k.dtype)
            
            v = tl.load(V_ptr)
            v = v * tl.exp(gv_last[None, :] - gv).to(v.dtype)

            if SAVE_S:
                tl.store(S_ptr, acc.to(S_ptr.dtype.element_ty))

            acc *= tl.exp(gk_last[:, None]).to(acc.dtype)
            acc *= tl.exp(gv_last[None, :]).to(acc.dtype)
            acc += tl.dot(k, v, allow_tf32=False).to(acc.dtype)
        

        K_ptr += D_MODEL_K * CHUNK_SIZE     
        Q_ptr += D_MODEL_K * CHUNK_SIZE
        V_ptr += D_MODEL_V * CHUNK_SIZE
        O_ptr += D_MODEL_V * CHUNK_SIZE 
        GV_ptr += D_MODEL_V * CHUNK_SIZE
        GV_last_ptr += D_MODEL_V * CHUNK_SIZE
        GK_ptr += D_MODEL_K * CHUNK_SIZE 
        GK_last_ptr += D_MODEL_K * CHUNK_SIZE
        S_ptr += D_MODEL_K * D_MODEL_V
    


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
        S = torch.empty(q.shape[0], q.shape[1], q.shape[2], k.shape[-1], v.shape[-1], device=q.device, dtype=q.dtype)
        # , memory_format=torch.contiguous_format)

        # o = torch.empty_like(v).contiguous()
        # share memory's limit.
        BLOCK_MODEL_K = 128
        BLOCK_MODEL_V = 128
 
        #split k
        o = torch.empty(B, H, D_k // BLOCK_MODEL_K, NUM_CHUNK, CHUNK_SIZE, D_v, device=q.device, dtype=q.dtype)
        # memory_format=torch.contiguous_format)
    

        assert D_k % BLOCK_MODEL_K == 0
        assert D_v % BLOCK_MODEL_V == 0

        grid = (B * H,  D_k//BLOCK_MODEL_K, D_v//BLOCK_MODEL_V)
        ctx.grid = grid 

        gk = gk 
        gv = gv

        gk.cumsum_(-2)
        gv.cumsum_(-2)
        
        _fwd_recurrence[grid](
            S, gk, gv, q, k, v, o,
            CHUNK_SIZE=CHUNK_SIZE, NUM_CHUNK = NUM_CHUNK, L = CHUNK_SIZE * NUM_CHUNK,
            D_MODEL_K=D_k, D_MODEL_V=D_v,
            BLOCK_MODEL_K=BLOCK_MODEL_K, BLOCK_MODEL_V=BLOCK_MODEL_V, SAVE_S=False, num_warps=16, num_stages=2
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
        raise NotImplementedError

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




def inter_chunk_onc_fused(query, key, value, gk, gv, chunk_size):
    L = query.shape[-2]
    num_block = L // chunk_size
    query = rearrange(query, 'b h (n c) d  -> b h n c d', c=chunk_size).contiguous()
    key = rearrange(key, 'b h (n c) d -> b h n c d', c=chunk_size).contiguous()
    value = rearrange(value, 'b h (n c) d -> b h n c d', c=chunk_size).contiguous()
    gk = rearrange(gk,  'b h (n c) d -> b h n c d', c=chunk_size).contiguous()
    gv = rearrange(gv,  'b h (n c) d -> b h n c d', c=chunk_size).contiguous()

    # gk = gk.cumsum(-2)
    # gv = gv.cumsum(-2)
    #### inter reduction
    inter_chunk_contribution = InterChunkFused.apply(query, key, value, gk, gv)
    
    return rearrange(inter_chunk_contribution, 'b h n c d -> b h (n c) d')





