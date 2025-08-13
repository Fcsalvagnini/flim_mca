import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class SaliencyFusionModel(nn.Module):
  def __init__(self, in_saliencies=4, in_channels=3):
    super().__init__()
    self.conv_o = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
    self.conv_s = nn.Conv2d(in_saliencies, 1, kernel_size=3, padding=1)

    self.head = nn.Conv2d(2, 1, kernel_size=1)
    self.sigmoid = nn.Sigmoid()

    self._initialize_weights()

  def _initialize_weights(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        # Glorot initialization
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
          # Initialize biases as zeros
          init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
          init.constant_(m.weight, 1)
          init.constant_(m.bias, 0)

  def forward(self, o, s):
    o = self.sigmoid(self.conv_o(o))
    s = self.sigmoid(self.conv_s(s))

    # Final prediction
    out = self.sigmoid(self.head(torch.cat([o, s], dim=1)))

    return out