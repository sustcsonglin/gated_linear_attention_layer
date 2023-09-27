import torch 
import os 
from torch.utils.cpp_extension import load
module_path = os.path.dirname(__file__)



cuda_compute_inner = load(
    name="cuda_compute_inner",
    sources=[os.path.join(module_path, "kernel_chunk16_dim64x.cpp"), os.path.join(module_path, "kernel_chunk16_dim64x.cu")],
    # extra_cuda_cflags=["-arch=sm_70"],  # Set the right compute capability based on your GPU
    verbose=True,
)



class CUDA_inner(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, gk, gv):
        original_dtype = q.dtype

        assert q.shape[-1] % 64 == 0
        assert q.shape[-2] == 16

        q = q.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()
        gk = gk.float().contiguous()
        gv = gv.float().contiguous()
    
        output, qk = cuda_compute_inner.forward(q, k, v, gk, gv)
        ctx.save_for_backward(q, k, v, gk, gv, qk)
        ctx.orig_dtype = original_dtype        
        
        return output.to(original_dtype), qk.to(original_dtype)



    @staticmethod
    def backward(ctx, do, dqk=None):
        orig_dtype =  ctx.orig_dtype
        do = do.float().contiguous()

        q, k, v, gk, gv, qk = ctx.saved_tensors
        dq, dk, dv, dgk, dgv = cuda_compute_inner.backward(q, k, v, gk, gv, qk, do)
        
        return dq.to(orig_dtype), dk.to(orig_dtype), dv.to(orig_dtype),  dgk.to(orig_dtype), dgv.to(orig_dtype)







cuda_compute_intra = CUDA_inner.apply




        
        

