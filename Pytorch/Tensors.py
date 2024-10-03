import torch
import numpy as np

data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

np_array = np.array(data)
x_np = torch.from_numpy(np_array)

x_ones = torch.ones_like(x_data)
print(f"Ones tensor \n {x_ones}")

x_rand = torch.rand_like(x_data, dtype = torch.float)
print(f"Random tensor \n {x_rand}")

shape = (2, 3)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)


print(f"Random \n {rand_tensor}")
print(f"Ones Tensor \n {ones_tensor}")
print(f"Zeros Tensor \n {zeros_tensor}")

tensor = torch.rand(shape)
print(f"Shape of tensor {tensor.shape}")
print(f"Data type of tensor {tensor.dtype}")
print(f"Device tensor is stored in, {tensor.device}")


tensor = torch.ones(4, 4)
print(f"First row {tensor[0]}")
print(f"First Column {tensor[:, 0]}")
print(f"Last column {tensor[..., -1]}")
