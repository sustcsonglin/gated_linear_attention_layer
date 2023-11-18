from triton_on2.full_no_fuse import compute_inter_chunk_on2
from triton_on2.full_fused import compute_inter_chunk_on2_fused


from triton_onc.chunk_scan_triton_full import inter_chunk_onc
from triton_on2.naive import naive
import torch 
import torch.nn.functional as F
from einops import rearrange
import time 
import math
from triton_onc.fused_inter_chunk_full import inter_chunk_onc_fused


if __name__ == "__main__":
    print("Hello...")
    B = 32
    H = 8
    L = 2048
    D_QK = 128
    D_V = 128
    require_grad= False



    dtype = torch.bfloat16
    q = torch.rand(B, H, L, D_QK, device='cuda').to(dtype).requires_grad_(require_grad)
    k = torch.rand(B, H, L, D_QK, device='cuda').to(dtype).requires_grad_(require_grad)
    v = torch.rand(B, H, L, D_V, device='cuda').to(dtype).requires_grad_(require_grad)
    gk = torch.randn(B, H, L, D_QK, device='cuda').to(dtype)
    gv = torch.randn(B, H, L, D_V, device='cuda').to(dtype)

    gk = (F.logsigmoid(gk) / 32).requires_grad_(require_grad)
    gv = (F.logsigmoid(gv) / 32).requires_grad_(require_grad)


    # o = inter_chunk_onc(q, k, v, gk, gv, chunk_size = 64)
    # # o2 = compute_inter_chunk_on2(q, k, v, gk, gv, chunk_size = 64)
    # o.sum().backward(retain_graph=True)
    
    # target = [q, k, v, gk, gv]
    # grad1= []
    # grad2= []
    # for s in target:
    #     grad1.append(s.grad.clone())
    #     s.grad.zero_()
    
    # o2 = inter_chunk_onc_fused(q, k, v, gk, gv, chunk_size = 64)
    # o2.sum().backward(retain_graph=True)

    # for s in target:
    #     grad2.append(s.grad.clone())
    #     s.grad.zero_()
    
    # # print( (o - o2).abs().max())

    # for a, b in zip(grad1, grad2):
    #     print( (a  - b).abs().max())


    for _ in range(100):
        o = inter_chunk_onc(q, k, v, gk, gv, chunk_size = 64)
        o3 = compute_inter_chunk_on2_fused(q, k, v, gk, gv, chunk_size = 64)
    
    print("Warmup.")

    start = time.time()
    
    for _ in range(200):
        o2 = inter_chunk_onc(q, k, v, gk, gv, chunk_size = 64)
        if require_grad:
            o2.sum().backward(retain_graph=True)    
        
    torch.cuda.synchronize()
    end = time.time()
    print(f"scan gret onc, time:{end - start}, ")      
    
    start = time.time()
    
    for _ in range(200):
        o3 = compute_inter_chunk_on2_fused(q, k, v, gk, gv, chunk_size = 64)
        if require_grad:
            o2.sum().backward(retain_graph=True)    
    torch.cuda.synchronize()
    end = time.time()
    print(f"scan gret onc fused, time:{end - start},  ")      
        

    print("warm up done.")



    # for chunk_size in [32, 64, 128, 256, 512, 1024, 2048]:
    #     torch.cuda.synchronize()
    #     start = time.time()    
    #     for _ in range(200):
    #         o2 = inter_chunk_onc(q, k, v, gk, gv, chunk_size = chunk_size)
    #         if require_grad:
    #             o2.sum().backward(retain_graph=True)    
    #     torch.cuda.synchronize()
    #     end = time.time()
    #     print(f"scan gret onc, time:{end - start},  chunk size: {chunk_size}")      
    # torch.cuda.synchronize()
    # start = time.time()
    # torch.cuda.synchronize()
    # start = time.time()


    # for _ in range(200):
    #     o = (q @ k.transpose(-1, -2)) @ v
    #     # if requires_grad:
    #     #     output2.sum().backward(retain_graph=True)    
    # torch.cuda.synchronize()
    # end = time.time()
    # print("flash gret on2, require_grad:{}", end - start)








    