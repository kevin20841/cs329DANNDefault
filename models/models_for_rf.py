import torch.nn as nn
from torchvision import models
from torch.autograd import Function

from receptivefield.pytorch import PytorchReceptiveField
import matplotlib.pyplot as plt
class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class Linear(nn.Module):
    def forward(self, x):
        return x

class MNISTmodel(nn.Module):
    """ MNIST architecture
    +Dropout2d, 84% ~ 73%
    -Dropout2d, 50% ~ 73%
    """

    def __init__(self):
        super(MNISTmodel, self).__init__()
        self.restored = False

        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32,
                      kernel_size=(3, 3)),  # 3 28 28, 32 24 24
            nn.BatchNorm2d(32),
            Linear(),
            nn.AvgPool2d(kernel_size=(2, 2)),  # 32 12 12
            nn.Conv2d(in_channels=32, out_channels=48,
                      kernel_size=(3, 3)),  # 48 8 8
            nn.BatchNorm2d(48),
            Linear(),
            nn.AvgPool2d(kernel_size=(2, 2)),  # 48 4 4
        )
    def forward(self, x):
        select = [4, 7]
        self.feature_maps = []
        for l, layer in enumerate(self.feature):
            x = layer(x)
            if l in select:
                self.feature_maps.append(x)
        return x
def model_fn() -> nn.Module:
    model = MNISTmodel()
    model.eval()
    return model


input_shape = [28, 28, 3]
rf = PytorchReceptiveField(model_fn)
rf_params = rf.compute(input_shape = input_shape)
from receptivefield.image import get_default_image
rf.plot_rf_grids(get_default_image(input_shape, name='cat'), figsize=(20,12),layout=(1, 2))

plt.savefig("field_mnist_default_architecture.png")
