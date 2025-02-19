import torch
import torch.nn as nn  

class SlicedWassersteinLoss(nn.Module):
  def __init__(self, nb_angles = 100, W_order=1, p_order =1, device = "cpu") -> None:
    super().__init__()
    self.nb_angles = nb_angles
    self.d = W_order
    self.device = device
    self.p_order = p_order

  def forward(self, input, target, proba):
    """
    - input: (batch, nb_points, 3)
    - target: (batch, nb_points, 3) 
    """

    angles = (torch.rand(1, 1, self.nb_angles, 3)*2-1).to(self.device)
    angles = angles/torch.linalg.vector_norm(angles, dim=-1, keepdim=True)

    inner_prod_input = (input.unsqueeze(-2)*angles).sum(dim=-1)
    inner_prod_target = (target.unsqueeze(-2)*angles).sum(dim=-1)

    sorted_input = torch.sort(inner_prod_input, dim=-2)[0]
    sorted_target = torch.sort(inner_prod_target, dim=-2)[0]

    diff = (sorted_input - sorted_target).abs()
    diff_pow = diff.pow(self.d)
    wasser = diff_pow.mean(dim=-2).pow(1/self.d).mean(dim=1)
    if self.p_order == 1:
      return (wasser*proba).sum()/(proba.sum())

    prob = proba.pow(self.p_order) 
    return (wasser*prob).sum()/(prob.sum())