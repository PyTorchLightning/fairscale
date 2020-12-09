import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import cast

import fairscale

model = nn.Sequential(torch.nn.Linear(10, 10), torch.nn.ReLU(), torch.nn.Linear(10, 5))
target = torch.randint(0, 2, size=(20, 1)).squeeze()
data = torch.randn(20, 10)
loss_fn = F.nll_loss

pipe_model = fairscale.nn.Pipe(model, balance=[2, 1])

# define optimizer and loss function
optimizer = optim.SGD(pipe_model.parameters(), lr=0.001)


# zero the parameter gradients
optimizer.zero_grad()

assert pipe_model.devices
device = pipe_model.devices[0]

# outputs and target need to be on the same device
# forward step
outputs = cast(torch.Tensor, pipe_model(data.to(device)))
# compute loss
loss = loss_fn(outputs.to(device), target.to(device))

# backward + optimize
loss.backward()
optimizer.step()

print("Finished Training Step")

del pipe_model
