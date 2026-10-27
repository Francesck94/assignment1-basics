import torch
from einops import rearrange, einsum
## Basic implementation

D = torch.randn(2, 3, 4)  # Example shape: batch=2, sequence=3, d_in=4
A = torch.randn(5, 4)     # Example shape: d_out=5, d_in=4
Y = D @ A.T
# Hard to tell the input and output shapes and what they mean.
# What shapes can D and A have, and do any of these have unexpected behavior?
## Einsum is self-documenting and robust
# D A -> Y
Y = einsum(D, A, "batch sequence d_in, d_out d_in -> batch sequence d_out")
## Or, a batched version where D can have any leading dimensions but A is constrained.
Y = einsum(D, A, "... d_in, d_out d_in -> ... d_out")

print(D.shape)
print(A.shape)
print(Y.shape)