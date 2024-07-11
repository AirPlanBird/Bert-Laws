import mindspore
from mindspore import nn


class TestNet(nn.Cell):
    def __init__(self):
        super(TestNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

    def construct(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        return x


net = TestNet()
# x = net()
# print(x.shape)
print(net)
