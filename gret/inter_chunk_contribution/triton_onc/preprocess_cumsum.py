import triton
import triton.language as tl
import torch
import torch.nn.functional as F
import time


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
        gk = tl.sigmoid(gk)
        gk = tl.log(gk + 1e-9) / normalizer
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
        gv = tl.sigmoid(gv)
        gv = tl.log(gv + 1e-9) / normalizer
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
        return gk_cumsum, gv_cumsum, k_reduce, v_reduce, q_exp, gv_cumsum_exp, gk_last_exp, gv_last_exp


    @staticmethod
    def backward(dq):
        raise NotImplementedError("PreprocessCumSum backward is not implemented")




def prepare_cumsum(query, key, value, g_key, g_value):
    g_key = g_key.float()
    g_value = g_value.float()
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
    requires_grad = False
    dtype = torch.bfloat16

    v1 = (torch.randn(B, H, num_chunk, chunk_size, D_K)).cuda().to(dtype).requires_grad_(requires_grad)
    v2 = (torch.randn(B, H, num_chunk, chunk_size, D_V)).cuda().to(dtype).requires_grad_(requires_grad) 
    g1 = torch.randn(B, H,  num_chunk, chunk_size, D_K).cuda().to(dtype).uniform_(0.9, 0.99).log().requires_grad_(requires_grad)
    g2 = torch.randn(B, H, num_chunk, chunk_size, D_V).cuda().to(dtype).uniform_(0.9, 0.99).log().requires_grad_(requires_grad)
    q = (torch.randn(B, H, num_chunk, chunk_size, D_K)).cuda().to(dtype).requires_grad_(requires_grad) 
    for _ in range(100):
        s = PreprocessCumSum.apply(q, v1, v2, g1, g2)
        s2 = prepare_cumsum(q, v1, v2, g1, g2)
    
    for ss, ss2 in zip(s, s2):
        print((ss-ss2).abs().max())
    
    print("Warmup.")
    # print('warm up done')
    torch.cuda.synchronize()

    start = time.time()

    for _ in range(1000):
        s = PreprocessCumSum.apply(q, v1, v2, g1, g2)
    
    torch.cuda.synchronize()
    end = time.time()
    print("Triton time: ", end - start)

    torch.cuda.synchronize()

    start = time.time()

    for _ in range(1000):
        s = prepare_cumsum(q, v1, v2, g1, g2)
    
    torch.cuda.synchronize()
    end = time.time()
    print("Pytorch time: ", end - start)






    
