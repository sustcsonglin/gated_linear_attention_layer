import triton
import triton.language as tl
import torch
import torch.nn.functional as F
import time

# def stable_logsigmoid(x):
#     # Use the identity log(sigmoid(x)) = -log(1 + exp(-x))
#     # This is stable for large negative values of x
#     neg_abs_x = -torch.abs(x)
#     return torch.where(x < 0, x, neg_abs_x) - torch.log1p(torch.exp(neg_abs_x))

@triton.jit 
def stable_log_sigmoid(x):
    neg_abs_x = -tl.where(x>0, x, -x)
    return tl.where(x < 0, x, neg_abs_x) - tl.log(1 + tl.exp(neg_abs_x))

@triton.jit
def _fwd_preprocess_cumsum_gk(
    Q, K, GK, GK_cumsum, 
    Q_exp, K_reduce, GK_last_exp, 
    NUM_CHUNK, L, normalizer,
    D_MODEL_K: tl.constexpr, 
    CHUNK_SIZE: tl.constexpr, 
  ):

    offset_bh = tl.program_id(0)
    offset_c = tl.program_id(1)
    Q_ptr = Q + offset_bh * L * D_MODEL_K + offset_c * CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_MODEL_K)
    Q_exp_ptr = Q_exp + offset_bh * L * D_MODEL_K + offset_c * CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_MODEL_K)

    GK_ptr = GK + offset_bh * L * D_MODEL_K + offset_c * CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_MODEL_K)
    
    GK_cumsum_ptr = GK_cumsum + offset_bh * L * D_MODEL_K + offset_c * CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_MODEL_K)

    GK_last_exp_ptr = GK_last_exp +  offset_bh * NUM_CHUNK * D_MODEL_K + offset_c * D_MODEL_K + tl.arange(0, D_MODEL_K)

    cumsum = tl.zeros([D_MODEL_K], dtype=tl.float32)

    for _ in range(CHUNK_SIZE):
        gk = tl.load(GK_ptr).to(tl.float32) 
        gk = stable_log_sigmoid(gk) / normalizer

        cumsum += gk 
        tl.store(GK_cumsum_ptr, cumsum.to(GK_cumsum_ptr.dtype.element_ty))

        cumsum_exp = tl.math.exp(cumsum)
        
        q = tl.load(Q_ptr)        
        q_exp = q * cumsum_exp
        tl.store(Q_exp_ptr, q_exp)

        Q_ptr += D_MODEL_K
        Q_exp_ptr += D_MODEL_K
        GK_ptr += D_MODEL_K
        GK_cumsum_ptr += D_MODEL_K

    tl.store(GK_last_exp_ptr, tl.math.exp(cumsum).to(GK_last_exp_ptr.dtype.element_ty))

    tl.debug_barrier()
    
    GK_cumsum_ptr = GK_cumsum + offset_bh * L * D_MODEL_K + offset_c * CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_MODEL_K)
    K_ptr = K + offset_bh * L * D_MODEL_K + offset_c * CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_MODEL_K)
    K_reduce_ptr = K_reduce + offset_bh * L * D_MODEL_K + offset_c * CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_MODEL_K)

    for _ in range(CHUNK_SIZE):
        gk_cumsum = tl.load(GK_cumsum_ptr)
        k = tl.load(K_ptr)
        k_reduce = k * tl.math.exp(cumsum - gk_cumsum)
        tl.store(K_reduce_ptr, k_reduce.to(K_reduce_ptr.dtype.element_ty))

        K_ptr += D_MODEL_K
        GK_cumsum_ptr += D_MODEL_K
        K_reduce_ptr += D_MODEL_K




@triton.jit
def _bwd_preprocess_cumsum_gk(
    Q, K, GK, GK_cumsum, 
    
    DQ_exp, DK_reduce, DGK_last_exp, DGK_cumsum, 

    DQ, DK, DGK, 

    NUM_CHUNK, L, normalizer,
    D_MODEL_K: tl.constexpr, 
    CHUNK_SIZE: tl.constexpr, 
  ):

    offset_bh = tl.program_id(0)
    offset_c = tl.program_id(1)
    Q_ptr = Q + offset_bh * L * D_MODEL_K + offset_c * CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_MODEL_K)
    K_ptr = K + offset_bh * L * D_MODEL_K + offset_c * CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_MODEL_K)
    GK_ptr = GK + offset_bh * L * D_MODEL_K + offset_c * CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_MODEL_K)
    GK_cumsum_ptr = GK_cumsum + offset_bh * L * D_MODEL_K + offset_c * CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_MODEL_K)

    DQ_ptr = DQ + offset_bh * L * D_MODEL_K + offset_c * CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_MODEL_K)
    DK_ptr = DK + offset_bh * L * D_MODEL_K + offset_c * CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_MODEL_K)
    DQ_exp_ptr = DQ_exp + offset_bh * L * D_MODEL_K + offset_c * CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_MODEL_K)
    DK_reduce_ptr = DK_reduce + offset_bh * L * D_MODEL_K + offset_c * CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_MODEL_K)
    DGK_cumsum_ptr = DGK_cumsum + offset_bh * L * D_MODEL_K + offset_c * CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_MODEL_K)
    DGK_ptr = DGK + offset_bh * L * D_MODEL_K + offset_c * CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_MODEL_K)

    D_GK_last_exp_ptr = DGK_last_exp + offset_bh * NUM_CHUNK * D_MODEL_K + offset_c * D_MODEL_K + tl.arange(0, D_MODEL_K) 
    # 
    cumsum_gradient = tl.zeros([D_MODEL_K], dtype=tl.float32)
    grad_gk_last = tl.zeros([D_MODEL_K], dtype=tl.float32)

    gk_last = tl.load(GK_cumsum_ptr + (CHUNK_SIZE - 1) * D_MODEL_K)    
    cumsum_gradient += tl.load(D_GK_last_exp_ptr) * tl.exp(gk_last)
    
    GK_ptr += (CHUNK_SIZE - 1) * D_MODEL_K
    GK_cumsum_ptr += (CHUNK_SIZE - 1) * D_MODEL_K
    Q_ptr += (CHUNK_SIZE - 1) * D_MODEL_K
    K_ptr += (CHUNK_SIZE - 1) * D_MODEL_K
    
    DQ_exp_ptr += (CHUNK_SIZE - 1) * D_MODEL_K
    DK_reduce_ptr += (CHUNK_SIZE - 1) * D_MODEL_K
    DGK_cumsum_ptr += (CHUNK_SIZE - 1) * D_MODEL_K
    DQ_ptr += (CHUNK_SIZE - 1) * D_MODEL_K
    DK_ptr += (CHUNK_SIZE - 1) * D_MODEL_K
    DGK_ptr += (CHUNK_SIZE - 1) * D_MODEL_K

    
    for idx in range(CHUNK_SIZE -1, -1, -1):
        gk_cs = tl.load(GK_cumsum_ptr).to(tl.float32)
        k = tl.load(K_ptr).to(tl.float32)
        grad_k = tl.exp(gk_last - gk_cs) * tl.load(DK_reduce_ptr).to(tl.float32)
        tl.store(DK_ptr, grad_k.to(DK_ptr.dtype.element_ty))
        grad_k *= k     
        cumsum_gradient -=  grad_k
        grad_gk_last += grad_k

        q = tl.load(Q_ptr).to(tl.float32)
        grad_q = tl.exp(gk_cs) * tl.load(DQ_exp_ptr) 
        tl.store(DQ_ptr, grad_q.to(DK_ptr.dtype.element_ty))
        cumsum_gradient += grad_q * q.to(tl.float32)

        # from intra-chunk contribution.
        cumsum_gradient += tl.load(DGK_cumsum_ptr).to(tl.float32) 
        
        tl.store(DGK_ptr, cumsum_gradient.to(DGK_ptr.dtype.element_ty))

        Q_ptr -= D_MODEL_K
        DQ_exp_ptr -= D_MODEL_K
        K_ptr -= D_MODEL_K
        DK_reduce_ptr -= D_MODEL_K
        GK_cumsum_ptr -= D_MODEL_K
        DGK_cumsum_ptr -= D_MODEL_K
        DQ_ptr -= D_MODEL_K
        DK_ptr -= D_MODEL_K
        DGK_ptr -= D_MODEL_K
    

    DGK_ptr =  DGK + offset_bh * L * D_MODEL_K + offset_c * CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_MODEL_K) + (CHUNK_SIZE - 1) * D_MODEL_K
    GK_ptr =  GK + offset_bh * L * D_MODEL_K + offset_c * CHUNK_SIZE * D_MODEL_K + tl.arange(0, D_MODEL_K) + (CHUNK_SIZE - 1) * D_MODEL_K

    # tl.store(D_GK_last_exp_ptr, cumsum_gradient)


    ## feel like i can do this without loop.
    # avoid strange bug....

    grad_gk_last = grad_gk_last + 0.
    for idx in range(CHUNK_SIZE -1, -1, -1):        
        dgk = tl.load(DGK_ptr).to(tl.float32)
        dgk += grad_gk_last
        
        gk = tl.load(GK_ptr).to(tl.float32) 
        gk = tl.sigmoid(gk)    
        dgk = (dgk / normalizer) * (1 - gk)
        tl.store(DGK_ptr, dgk.to(DGK_ptr.dtype.element_ty))
        DGK_ptr -= D_MODEL_K
        GK_ptr -= D_MODEL_K
    


    







@triton.jit
def _fwd_preprocess_cumsum_gv(
    V, GV,  
    GV_cumsum, GV_exp, V_reduce, GV_last_exp, 
    NUM_CHUNK, L, normalizer,
    D_MODEL_V: tl.constexpr, 
    CHUNK_SIZE: tl.constexpr, 
  ):

    offset_bh = tl.program_id(0)
    offset_c = tl.program_id(1)

    GV_ptr = GV + offset_bh * L * D_MODEL_V + offset_c * CHUNK_SIZE * D_MODEL_V + tl.arange(0, D_MODEL_V)
    
    GV_last_exp_ptr = GV_last_exp + offset_bh * NUM_CHUNK * D_MODEL_V + offset_c * D_MODEL_V + tl.arange(0, D_MODEL_V)

    GV_cumsum_ptr = GV_cumsum + offset_bh * L * D_MODEL_V + offset_c * CHUNK_SIZE * D_MODEL_V + tl.arange(0, D_MODEL_V)
    GV_exp_ptr = GV_exp + offset_bh * L * D_MODEL_V + offset_c * CHUNK_SIZE * D_MODEL_V + tl.arange(0, D_MODEL_V)

    cumsum = tl.zeros([D_MODEL_V], dtype=tl.float32)
    
    for _ in range(CHUNK_SIZE):
        gv = tl.load(GV_ptr).to(tl.float32) 
        # gv = tl.sigmoid(gv)
        # gv = tl.log(gv + 1e-9) / normalizer
        gv = stable_log_sigmoid(gv) / normalizer
        cumsum += gv

        tl.store(GV_cumsum_ptr, cumsum.to(GV_cumsum_ptr.dtype.element_ty))
        tl.store(GV_exp_ptr, tl.math.exp(cumsum).to(GV_cumsum_ptr.dtype.element_ty))
        
        GV_cumsum_ptr += D_MODEL_V
        GV_exp_ptr += D_MODEL_V
        GV_ptr += D_MODEL_V

    tl.store(GV_last_exp_ptr, tl.math.exp(cumsum).to(GV_last_exp_ptr.dtype.element_ty))
    
    tl.debug_barrier()
    
    V_ptr = V + offset_bh * L * D_MODEL_V + offset_c * CHUNK_SIZE * D_MODEL_V + tl.arange(0, D_MODEL_V)    
    GV_cumsum_ptr = GV_cumsum + offset_bh * L * D_MODEL_V + offset_c * CHUNK_SIZE * D_MODEL_V + tl.arange(0, D_MODEL_V)
    V_reduce_ptr = V_reduce + offset_bh * L * D_MODEL_V + offset_c * CHUNK_SIZE * D_MODEL_V + tl.arange(0, D_MODEL_V)    

    for _ in range(CHUNK_SIZE):
        v = tl.load(V_ptr)                
        gv = tl.load(GV_cumsum_ptr)
        v_reduce = v * tl.math.exp(cumsum - gv)
        tl.store(V_reduce_ptr, v_reduce.to(V_reduce_ptr.dtype.element_ty))
        
        V_ptr += D_MODEL_V
        V_reduce_ptr += D_MODEL_V
        GV_cumsum_ptr += D_MODEL_V
    
@triton.jit
def _bwd_preprocess_cumsum_gv(
    V, GV, GV_cumsum,     

    DGV_cumsum_exp, DV_reduce, DGV_last_exp, DGV_cumsum, 
    DV, DGV, 

    NUM_CHUNK, L, normalizer,
    D_MODEL_V: tl.constexpr, 
    CHUNK_SIZE: tl.constexpr, 
  ):

    offset_bh = tl.program_id(0)
    offset_c = tl.program_id(1)
    V_ptr = V + offset_bh * L * D_MODEL_V + offset_c * CHUNK_SIZE * D_MODEL_V + tl.arange(0, D_MODEL_V)
    GV_ptr = GV + offset_bh * L * D_MODEL_V + offset_c * CHUNK_SIZE * D_MODEL_V + tl.arange(0, D_MODEL_V)
    GV_cumsum_ptr = GV_cumsum + offset_bh * L * D_MODEL_V + offset_c * CHUNK_SIZE * D_MODEL_V + tl.arange(0, D_MODEL_V)

    DV_ptr = DV + offset_bh * L * D_MODEL_V + offset_c * CHUNK_SIZE * D_MODEL_V + tl.arange(0, D_MODEL_V)
    DV_reduce_ptr = DV_reduce + offset_bh * L * D_MODEL_V + offset_c * CHUNK_SIZE * D_MODEL_V + tl.arange(0, D_MODEL_V)
    DGV_cumsum_ptr = DGV_cumsum + offset_bh * L * D_MODEL_V + offset_c * CHUNK_SIZE * D_MODEL_V + tl.arange(0, D_MODEL_V)
    DGV_cumsum_exp_ptr = DGV_cumsum_exp + offset_bh * L * D_MODEL_V + offset_c * CHUNK_SIZE * D_MODEL_V + tl.arange(0, D_MODEL_V)

    DGV_ptr = DGV + offset_bh * L * D_MODEL_V + offset_c * CHUNK_SIZE * D_MODEL_V + tl.arange(0, D_MODEL_V)

    D_GV_last_exp_ptr = DGV_last_exp + offset_bh * NUM_CHUNK * D_MODEL_V + offset_c * D_MODEL_V + tl.arange(0, D_MODEL_V) 
     
    cumsum_gradient = tl.zeros([D_MODEL_V], dtype=tl.float32)
    grad_gv_last = tl.zeros([D_MODEL_V], dtype=tl.float32)

    gv_last = tl.load(GV_cumsum_ptr + (CHUNK_SIZE - 1) * D_MODEL_V)    
    cumsum_gradient += tl.load(D_GV_last_exp_ptr) * tl.exp(gv_last)
    
    GV_ptr += (CHUNK_SIZE - 1) * D_MODEL_V
    GV_cumsum_ptr += (CHUNK_SIZE - 1) * D_MODEL_V

    V_ptr += (CHUNK_SIZE - 1) * D_MODEL_V 
    
    DV_reduce_ptr += (CHUNK_SIZE - 1) * D_MODEL_V
    DGV_cumsum_ptr += (CHUNK_SIZE - 1) * D_MODEL_V
    DGV_cumsum_exp_ptr += (CHUNK_SIZE - 1) * D_MODEL_V
    DV_ptr += (CHUNK_SIZE - 1) * D_MODEL_V
    DGV_ptr += (CHUNK_SIZE - 1) * D_MODEL_V

    for idx in range(CHUNK_SIZE -1, -1, -1):
        gv_cs = tl.load(GV_cumsum_ptr).to(tl.float32)
        v = tl.load(V_ptr).to(tl.float32)
        grad_v = tl.exp(gv_last - gv_cs) * tl.load(DV_reduce_ptr).to(tl.float32)
        tl.store(DV_ptr, grad_v.to(DV_ptr.dtype.element_ty))
        grad_v *= v
        cumsum_gradient -= grad_v
        grad_gv_last += grad_v

        # q = tl.load(Q_ptr).to(tl.float32)
        grad_v = tl.exp(gv_cs) * tl.load(DGV_cumsum_exp_ptr) 
        cumsum_gradient += grad_v

        # from intra-chunk contribution.
        cumsum_gradient += tl.load(DGV_cumsum_ptr).to(tl.float32) 
        
        tl.store(DGV_ptr, cumsum_gradient.to(DGV_ptr.dtype.element_ty))

        V_ptr -= D_MODEL_V
        DV_reduce_ptr -= D_MODEL_V
        GV_cumsum_ptr -= D_MODEL_V
        DGV_cumsum_ptr -= D_MODEL_V
        DV_ptr -= D_MODEL_V
        DGV_ptr -= D_MODEL_V
        DGV_cumsum_exp_ptr -= D_MODEL_V
 
    DGV_ptr =  DGV + offset_bh * L * D_MODEL_V + offset_c * CHUNK_SIZE * D_MODEL_V + tl.arange(0, D_MODEL_V) + (CHUNK_SIZE - 1) * D_MODEL_V
    GV_ptr =  GV + offset_bh * L * D_MODEL_V + offset_c * CHUNK_SIZE * D_MODEL_V + tl.arange(0, D_MODEL_V) + (CHUNK_SIZE - 1) * D_MODEL_V
    
    grad_gv_last = grad_gv_last + 0.

    for idx in range(CHUNK_SIZE -1, -1, -1):        
        dgv = tl.load(DGV_ptr).to(tl.float32)
        dgv += grad_gv_last
        gv = tl.load(GV_ptr).to(tl.float32) 
        gv = tl.sigmoid(gv)    
        dgv = (dgv / normalizer) * (1 - gv)
        tl.store(DGV_ptr, dgv.to(DGV_ptr.dtype.element_ty))
        DGV_ptr -= D_MODEL_V
        GV_ptr -= D_MODEL_V
    


class PreprocessCumSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, gk, gv, normalizer_gk=8, normalizer_gv=8):
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        gk = gk.contiguous()
        gv = gv.contiguous()
        
        B, H, NUM_CHUNK, CHUNK_SIZE, D = q.shape

        D_k = k.shape[-1]
        D_v = v.shape[-1]
        
        # (B, H, L, D_K, D_V)
        # , memory_format=torch.contiguous_format)
        # o = torch.empty_like(v).contiguous()
        # share memory's limit.
        # BLOCK_MODEL_K = 128
        # BLOCK_MODEL_V = 128
        #split k

        grid = (B * H, NUM_CHUNK)
        ctx.grid = grid 

        gk = gk 
        gv = gv

        k_reduce = torch.empty_like(k)
        q_exp = torch.empty_like(q)

        gk_cumsum = torch.empty_like(gk, dtype=torch.float32)

        gk_last_exp = torch.empty_like(gk[:, :, :, 0], dtype=torch.float32)

        _fwd_preprocess_cumsum_gk[grid](
            q, k, gk, gk_cumsum, 
            q_exp, k_reduce, gk_last_exp, 
            CHUNK_SIZE=CHUNK_SIZE, NUM_CHUNK = NUM_CHUNK, L = CHUNK_SIZE * NUM_CHUNK, normalizer=8,
            D_MODEL_K=D_k, num_warps=8 if D_k >= 512 else 4
        )

        gv_cumsum = torch.empty_like(gv, dtype=torch.float32)                        
        gv_cumsum_exp = torch.empty_like(gv_cumsum, dtype=torch.float32)
        v_reduce = torch.empty_like(v)
        gv_last_exp = torch.empty_like(gv[:, :, :, 0], dtype=torch.float32)
        
        _fwd_preprocess_cumsum_gv[grid](
            v, gv,  gv_cumsum, gv_cumsum_exp,  
            v_reduce, gv_last_exp, 
            CHUNK_SIZE=CHUNK_SIZE, NUM_CHUNK = NUM_CHUNK, L = CHUNK_SIZE * NUM_CHUNK, normalizer=8,
            D_MODEL_V=D_v, num_warps=8 if D_v >= 512 else 4
        )

        ctx.grid = grid 
        ctx.save_for_backward(q, k, v, gk, gv, gk_cumsum, gv_cumsum)

        return gk_cumsum, gv_cumsum, k_reduce, v_reduce, q_exp, gv_cumsum_exp, gk_last_exp, gv_last_exp



    @staticmethod
    def backward(ctx, dgk_cumsum, dgv_cumsum, dk_reduce, dv_reduce, dq_exp, dgv_cumsum_exp, dgk_last_exp, dgv_last_exp):
        dgk_cumsum = dgk_cumsum.contiguous()
        dgv_cumsum = dgv_cumsum.contiguous()
        dk_reduce = dk_reduce.contiguous()
        dv_reduce = dv_reduce.contiguous()
        dq_exp = dq_exp.contiguous()
        dgv_cumsum_exp = dgv_cumsum_exp.contiguous()
        dgk_last_exp = dgk_last_exp.contiguous()
        dgv_last_exp = dgv_last_exp.contiguous()

        q, k, v, gk, gv, gk_cumsum, gv_cumsum = ctx.saved_tensors
        grid  = ctx.grid

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dgk = torch.empty_like(gk, dtype=torch.float32)

        B, H, NUM_CHUNK, CHUNK_SIZE, D_k = q.shape


        D_v = v.shape[-1]        

        _bwd_preprocess_cumsum_gk[grid](
            q, k, gk, gk_cumsum, 
            dq_exp, dk_reduce, dgk_last_exp, dgk_cumsum,
            dq, dk, dgk,
            CHUNK_SIZE=CHUNK_SIZE, NUM_CHUNK = NUM_CHUNK, L = CHUNK_SIZE * NUM_CHUNK, normalizer=8,
            D_MODEL_K=D_k, num_warps=8 if D_k >= 512 else 4
        )

        dv = torch.empty_like(v)
        dgv = torch.empty_like(gv, dtype=torch.float32)
        
        _bwd_preprocess_cumsum_gv[grid](
            v, gv, gv_cumsum,  dgv_cumsum_exp, dv_reduce, dgv_last_exp, dgv_cumsum, 
            dv, dgv, 
            CHUNK_SIZE=CHUNK_SIZE, NUM_CHUNK = NUM_CHUNK, L = CHUNK_SIZE * NUM_CHUNK, normalizer=8,
            D_MODEL_V=D_v, num_warps=8 if D_k >= 512 else 4 
        )
        
        return dq, dk, dv, dgk, dgv, None, None
    


def prepare_cumsum(query, key, value, g_key, g_value):
    g_key = g_key 
    g_value = g_value
    g_key = F.logsigmoid(g_key) / 8
    g_key_cumsum = g_key.cumsum(-2)
    reduce_chunk_key = (g_key_cumsum[..., -1, None, :] -  g_key_cumsum).exp()

    g_value = F.logsigmoid(g_value) / 8
    g_value_cumsum = g_value.cumsum(-2)
    reduce_chunk_value = (g_value_cumsum[..., -1, None, :] - g_value_cumsum).exp()
    
    reduce_value = value * reduce_chunk_value.to(value)
    reduce_key = key * reduce_chunk_key.to(key)    
    
    g_key_last_exp = g_key_cumsum[:, :, :, -1].exp()
    g_value_last_exp = g_value_cumsum[:, :, :, -1].exp()

    g_value_cumsum_exp = g_value_cumsum.exp().to(key)
    q_exp = query * g_key_cumsum.exp().to(query)

    return  g_key_cumsum, g_value_cumsum,  reduce_key, reduce_value, q_exp, g_value_cumsum_exp, g_key_last_exp, g_value_last_exp
    
    
    
        
    
# to_add = (key * reduce_chunk_key.to(key)).transpose(-1, -2) @  (value * reduce_chunk_value.to(value))
 
    
    

if __name__ == "__main__":
    B = 32
    H = 4
    L = 2048
    D_K = 256
    D_V = 512
    
    chunk_size = 32
    num_chunk = L // chunk_size
    device = "cuda"
    requires_grad = True
    dtype = torch.bfloat16
    
    v1 = (torch.randn(B, H, num_chunk, chunk_size, D_K)).cuda().to(dtype).requires_grad_(requires_grad)
    v2 = (torch.randn(B, H, num_chunk, chunk_size, D_V)).cuda().to(dtype).requires_grad_(requires_grad) 
    g1 = torch.randn(B, H,  num_chunk, chunk_size, D_K).cuda().to(dtype).uniform_(0.95, 0.99).log().requires_grad_(requires_grad)
    g2 = torch.randn(B, H, num_chunk, chunk_size, D_V).cuda().to(dtype).uniform_(0.95, 0.99).log().requires_grad_(requires_grad)
    q = (torch.randn(B, H, num_chunk, chunk_size, D_K)).cuda().to(dtype).requires_grad_(requires_grad) 

    test_speed= True 
    test_gradient = True

    if test_gradient:
        target = [v1, v2, g1, g2, q]
        grad1= [ ]
        g_key_cumsum, g_value_cumsum,  reduce_key, reduce_value, q_exp, g_value_cumsum_exp, g_key_last_exp, g_value_last_exp = PreprocessCumSum.apply(q, v1, v2, g1, g2)
        
        o = (g_key_cumsum ** 2).sum() + (g_value_cumsum ** 2).sum() + (reduce_key **2).sum() + (q_exp**2).sum() + (g_value_cumsum_exp **2).sum() + (g_key_last_exp**2).sum() + (g_value_last_exp **2).sum()
        o.sum().backward(retain_graph=True )
        
        for v in target:   
            grad1.append(v.grad.clone())
            v.grad.zero_()
        
        g_key_cumsum, g_value_cumsum,  reduce_key, reduce_value, q_exp, g_value_cumsum_exp, g_key_last_exp, g_value_last_exp = prepare_cumsum(q, v1, v2, g1, g2)
        o = (g_key_cumsum ** 2).sum() + (g_value_cumsum ** 2).sum() + (reduce_key **2).sum() + (q_exp**2).sum() + (g_value_cumsum_exp **2).sum() + (g_key_last_exp**2).sum() + (g_value_last_exp **2).sum()
        o.sum().backward(retain_graph=True )

        grad2= [ ]

        for v in target:
            grad2.append(v.grad.clone())
            v.grad.zero_()
        
        for ss1,ss2 in zip(grad1, grad2):
            print( (ss1 - ss2).abs().max())
    
            
        
  
        
    #### speed testing
    if test_speed:

        for _ in range(100):
            g_key_cumsum, g_value_cumsum,  reduce_key, reduce_value, q_exp, g_value_cumsum_exp, g_key_last_exp, g_value_last_exp = PreprocessCumSum.apply(q, v1, v2, g1, g2)

            if requires_grad:
                o = (g_key_cumsum ** 2).sum() + (g_value_cumsum ** 2).sum() + (reduce_key **2).sum() + (q_exp**2).sum() + (g_value_cumsum_exp **2).sum() + (g_key_last_exp**2).sum() + (g_value_last_exp **2).sum()
                o.sum().backward(retain_graph=True )

            g_key_cumsum, g_value_cumsum,  reduce_key, reduce_value, q_exp, g_value_cumsum_exp, g_key_last_exp, g_value_last_exp = prepare_cumsum(q, v1, v2, g1, g2)
            if requires_grad:
                o = (g_key_cumsum ** 2).sum() + (g_value_cumsum ** 2).sum() + (reduce_key **2).sum() + (q_exp**2).sum() + (g_value_cumsum_exp **2).sum() + (g_key_last_exp**2).sum() + (g_value_last_exp **2).sum()
                o.sum().backward(retain_graph=True )
        
        # for ss, ss2 in zip(s, s2):
        #     print((ss-ss2).abs().max())
        
        print("Warmup.")
        # print('warm up done')
        torch.cuda.synchronize()

        start = time.time()

        for _ in range(200):
            
            g_key_cumsum, g_value_cumsum,  reduce_key, reduce_value, q_exp, g_value_cumsum_exp, g_key_last_exp, g_value_last_exp = PreprocessCumSum.apply(q, v1, v2, g1, g2)
            
            if requires_grad:
                o = (g_key_cumsum ** 2).sum() + (g_value_cumsum ** 2).sum() + (reduce_key **2).sum() + (q_exp**2).sum() + (g_value_cumsum_exp **2).sum() + (g_key_last_exp**2).sum() + (g_value_last_exp **2).sum()
                o.sum().backward(retain_graph=True )

        
        torch.cuda.synchronize()
        end = time.time()
        print("Triton time: ", end - start)

        torch.cuda.synchronize()
        start = time.time()

        for _ in range(200):
            g_key_cumsum, g_value_cumsum,  reduce_key, reduce_value, q_exp, g_value_cumsum_exp, g_key_last_exp, g_value_last_exp = prepare_cumsum(q, v1, v2, g1, g2)
            if requires_grad:
                o = (g_key_cumsum ** 2).sum() + (g_value_cumsum ** 2).sum() + (reduce_key **2).sum() + (q_exp**2).sum() + (g_value_cumsum_exp **2).sum() + (g_key_last_exp**2).sum() + (g_value_last_exp **2).sum()
                o.sum().backward(retain_graph=True )
                
        torch.cuda.synchronize()
        end = time.time()
        print("Pytorch time: ", end - start)






        
