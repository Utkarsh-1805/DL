import torch

A = torch.tensor([[1,2],[3,4]], dtype=torch.float32)
B = torch.tensor([[5,6],[7,8]], dtype=torch.float32)

# Addition
print("A + B =\n", A + B)

# Subtraction
print("A - B =\n", A - B)

# Matrix Multiplication
print("A x B =\n", torch.matmul(A, B))

# Transpose
print("Transpose of A =\n", A.t())

# Determinant
print("Determinant of A =", torch.det(A))

# Inverse
print("Inverse of A =\n", torch.inverse(A))


