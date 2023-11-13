import torch

B = 4
L = 2048
D = 256
H = 4

q = torch.rand(B, H, L, D).cuda().to(torch.bfloat16)
k = torch.rand(B, H, L, D).cuda().to(torch.bfloat16)
import time 
torch.cuda.synchronize()
start = time.time()

for _ in range(1000):
    q2 = q.float()
    k2 = k.float()


torch.cuda.synchronize()
print(time.time() - start)


