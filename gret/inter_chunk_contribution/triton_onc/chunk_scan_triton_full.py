import torch
from einops import rearrange
import triton 
import triton.language as tl
import torch.nn as nn
import torch.nn.functional as F

@triton.jit
def _fwd_recurrence(
    S, p1, p2, 
    O,
    NUM_BLOCK, 
    D_MODEL_K: tl.constexpr, D_MODEL_V: tl.constexpr,
    BLOCK_MODEL: tl.constexpr
  ):
    offset_bh = tl.program_id(0)
    offset_d = tl.program_id(1)
    offset_s = tl.program_id(2)    
    
    S = S + offset_bh * NUM_BLOCK * D_MODEL_K * D_MODEL_V + offset_d * D_MODEL_V * BLOCK_MODEL  +  tl.arange(0, BLOCK_MODEL)[:, None] * D_MODEL_V + offset_s * BLOCK_MODEL + tl.arange(0, BLOCK_MODEL)[None, :]

    O = O + offset_bh * NUM_BLOCK * D_MODEL_K * D_MODEL_V + offset_d * D_MODEL_V * BLOCK_MODEL  +  tl.arange(0, BLOCK_MODEL)[:, None] * D_MODEL_V + offset_s * BLOCK_MODEL + tl.arange(0, BLOCK_MODEL)[None, :] +  D_MODEL_K * D_MODEL_V    

    p1 = p1 + offset_bh * NUM_BLOCK * D_MODEL_K + tl.arange(0, BLOCK_MODEL) + offset_d * BLOCK_MODEL + D_MODEL_K     

    p2 = p2 + offset_bh * NUM_BLOCK * D_MODEL_V + tl.arange(0, BLOCK_MODEL) + offset_s * BLOCK_MODEL + D_MODEL_V  

    acc = tl.zeros([BLOCK_MODEL, BLOCK_MODEL], dtype=tl.float32)
    acc += tl.load(S)    
    
    S += D_MODEL_K * D_MODEL_V    

    tl.store(O, acc.to(O.dtype.element_ty))
    O += D_MODEL_K * D_MODEL_V

    for i in range(NUM_BLOCK-2):
        p_k = tl.load(p1)
        p_v = tl.load(p2)
        S_i = tl.load(S) 
        acc = acc * p_k[:, None] * p_v[None, :] + S_i
        tl.store(O, acc.to(O.dtype.element_ty))
        p1 +=  D_MODEL_K
        p2 += D_MODEL_V
        S +=  D_MODEL_K * D_MODEL_V
        O +=  D_MODEL_K * D_MODEL_V       


## NUM_SPLIT_K/V. K/V dimension split into NUM_SPLIT_K/V parts with equal size BLOCK_MODEL
@triton.jit
def _bwd_recurrence(
    S, p1, p2, 
    DS, Dp1, Dp2, 

    NUM_BLOCK, NUM_SPLIT_K, NUM_SPLIT_V,
    D_MODEL_K: tl.constexpr, D_MODEL_V: tl.constexpr,
    BLOCK_MODEL: tl.constexpr
    
 ):
    offset_bh = tl.program_id(0)
    offset_d = tl.program_id(1)
    offset_s = tl.program_id(2)    

    # skip the last chunk because it is never used
    S = S + offset_bh * NUM_BLOCK * D_MODEL_K * D_MODEL_V + offset_d * D_MODEL_V * BLOCK_MODEL + tl.arange(0, BLOCK_MODEL)[:, None] * D_MODEL_V + offset_s * BLOCK_MODEL + tl.arange(0, BLOCK_MODEL)[None, :]  + (NUM_BLOCK - 2) * D_MODEL_K * D_MODEL_V

    # start from the last chunk  
    DS = DS + offset_bh * NUM_BLOCK * D_MODEL_K * D_MODEL_V + offset_d * D_MODEL_V * BLOCK_MODEL  +  tl.arange(0, BLOCK_MODEL)[:, None] * D_MODEL_V + offset_s * BLOCK_MODEL + tl.arange(0, BLOCK_MODEL)[None, :]  + (NUM_BLOCK - 1) * D_MODEL_K * D_MODEL_V

    # skip the last chunk because it is never used  
    p1 = p1 + offset_bh * NUM_BLOCK * D_MODEL_K + tl.arange(0, BLOCK_MODEL) + offset_d * BLOCK_MODEL + (NUM_BLOCK - 2) * D_MODEL_K 

    # skip the last chunk because it is never used 
    p2 = p2 + offset_bh * NUM_BLOCK * D_MODEL_V + tl.arange(0, BLOCK_MODEL) + offset_s * BLOCK_MODEL + (NUM_BLOCK - 2) * D_MODEL_V 

    # skip the last chunk because it is never used  
    # NUM_BLOCK * D_MODEL_K * NUM_SPLIT_V: stride_bh
    # offset_s * D_MODEL_K: find the right split in the K dimension
    Dp1 = Dp1 + offset_bh * NUM_BLOCK * D_MODEL_K * NUM_SPLIT_V + offset_s * D_MODEL_K + tl.arange(0, BLOCK_MODEL) + offset_d * BLOCK_MODEL + (NUM_BLOCK - 2) * D_MODEL_K * NUM_SPLIT_V


    # skip the last chunk because it is never used 
    Dp2 = Dp2 + offset_bh * NUM_BLOCK * D_MODEL_V * NUM_SPLIT_K + offset_d * D_MODEL_V + tl.arange(0, BLOCK_MODEL) + offset_s * BLOCK_MODEL + (NUM_BLOCK - 2) * D_MODEL_V  * NUM_SPLIT_K

    Dacc = tl.zeros([BLOCK_MODEL, BLOCK_MODEL], dtype=tl.float32) 

    # ignore the first chunk
    for i in range(NUM_BLOCK - 1):

        p_key = tl.load(p1)
        p_value = tl.load(p2)
        S_i = tl.load(S)
        DS_i = tl.load(DS)

        Dacc += DS_i         
        
        dp_i = Dacc * S_i
        
        dp_key = tl.sum(dp_i * p_value[None, :], axis=1)
        tl.store(Dp1, dp_key.to(Dp1.dtype.element_ty))
        dp_value = tl.sum(dp_i * p_key[:, None], axis=0) 
        tl.store(Dp2, dp_value.to(Dp2.dtype.element_ty))

        tl.store(S, Dacc.to(S.dtype.element_ty))        

        Dacc *= p_key[:, None]
        Dacc *= p_value[None, :]

        S -= D_MODEL_K * D_MODEL_V 
        DS -= D_MODEL_K * D_MODEL_V 
        p1 -= D_MODEL_K 
        p2 -= D_MODEL_V 
        Dp1 -= D_MODEL_K * NUM_SPLIT_V
        Dp2 -= D_MODEL_V * NUM_SPLIT_K
    


class Chunk_memory_update(torch.autograd.Function):
    @staticmethod
    def forward(ctx, decay_key_last, decay_value_last, to_add):
        decay_key_last = decay_key_last.contiguous()
        decay_value_last = decay_value_last.contiguous()
        to_add = to_add.contiguous()

        B, H, N, D_k, D_v = to_add.shape 
        output = torch.empty_like(to_add)        
        BLOCK_MODEL = 128
    
        assert D_k % 32 == 0
        assert D_v % 32 == 0
        assert D_k == decay_key_last.shape[-1]
        assert D_v == decay_value_last.shape[-1]

        grid = (B*H, D_k//BLOCK_MODEL, D_v//BLOCK_MODEL)
        ctx.grid = grid 
        ctx.BLOCK_MODEL = BLOCK_MODEL

        _fwd_recurrence[grid](
            to_add,  
            decay_key_last,
            decay_value_last,
            output,
            D_MODEL_K=D_k, D_MODEL_V=D_v,
            NUM_BLOCK=N,  
            BLOCK_MODEL=BLOCK_MODEL, num_warps=16, num_stages=8
        )
    

        
        output[:, :, 0] = 0
        ctx.save_for_backward(output, decay_key_last, decay_value_last)        
        
        return output

    @staticmethod
    def backward(ctx, DO):
        DO = DO.contiguous()

        output, decay_key_last, decay_value_last = ctx.saved_tensors 

        B, H, N, D_k, D_v = output.shape 

        num_block = N

        grid = ctx.grid 
        BLOCK_MODEL = ctx.BLOCK_MODEL 


        # I don't want atomic_add to be used in the backward pass
        # so I add another dimension to the output tensor (D_k/v // BLOCK_MODEL)
        # afterward, I sum over this dimension to get the correct gradient 
        D_p1 = torch.empty(B, H, N, D_v // BLOCK_MODEL, D_k, device=DO.device)
        D_p2 = torch.empty(B, H, N, D_k // BLOCK_MODEL, D_v, device=DO.device)

        _bwd_recurrence[grid](
            output, decay_key_last, decay_value_last,
            DO, D_p1, D_p2, 
            NUM_BLOCK = num_block, NUM_SPLIT_K = D_k // BLOCK_MODEL, NUM_SPLIT_V = D_v // BLOCK_MODEL, 
            D_MODEL_K = D_k,
            D_MODEL_V = D_v, 
            BLOCK_MODEL = BLOCK_MODEL, num_warps=16, num_stages=4
        )

        output[:, :, -1] = 0
        D_p1[:, :, 0] = 0
        D_p1[:, :, -1] = 0
        D_p2[:, :, 0] = 0
        D_p2[:, :, -1] = 0
        

        return D_p1.sum(-2), D_p2.sum(-2), output        




def inter_chunk_onc(query, key, value, gk, gv, chunk_size):
    L = query.shape[-2]
    num_block = L // chunk_size
    query = rearrange(query, 'b h (n c) d  -> b h n c d', c=chunk_size).contiguous()
    key = rearrange(key, 'b h (n c) d -> b h n c d', c=chunk_size).contiguous()
    value = rearrange(value, 'b h (n c) d -> b h n c d', c=chunk_size).contiguous()
    gk = rearrange(gk,  'b h (n c) d -> b h n c d', c=chunk_size).contiguous()
    gv = rearrange(gv,  'b h (n c) d -> b h n c d', c=chunk_size).contiguous()

    gk = gk.cumsum(-2)
    gv = gv.cumsum(-2)


    #### inter reduction
    reduce_chunk_key = (gk[..., -1, None, :] -  gk).exp().to(key)
    reduce_chunk_value = (gv[..., -1, None, :] -  gv).exp().to(key)
    decay_key_chunk = (gk[..., -1, :]).exp() 
    decay_value_chunk = (gv[..., -1, :]).exp() 
    to_add = (key * reduce_chunk_key).transpose(-1, -2) @ (value * reduce_chunk_value)
    
    
    #### inter scan
    memory_cache = Chunk_memory_update.apply(decay_key_chunk, decay_value_chunk, to_add)    

    ### inter contribution
    inter_chunk_contribution = ((query * gk.exp()) @ memory_cache) * gv.exp() 
    
    return rearrange(inter_chunk_contribution, 'b h n c d -> b h (n c) d')






    

    

    