import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.learning import freeze_params
from torchvision.transforms import functional as transformF
from torchvision.transforms import InterpolationMode


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 BatchNorm=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               dilation=dilation,
                               padding=dilation,
                               bias=False)
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, output_stride, BatchNorm, freeze_at=0):
        self.inplanes = 64
        super(ResNet, self).__init__()

        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            raise NotImplementedError
        self.strides = strides

        # Modules
        self.conv1 = nn.Conv2d(3,
                               64,
                               kernel_size=7,
                               stride=2,
                               padding=3,
                               bias=False)
        self.bn1 = BatchNorm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block,
                                       64,
                                       layers[0],
                                       stride=strides[0],
                                       dilation=dilations[0],
                                       BatchNorm=BatchNorm)
        self.layer2 = self._make_layer(block,
                                       128,
                                       layers[1],
                                       stride=strides[1],
                                       dilation=dilations[1],
                                       BatchNorm=BatchNorm)
        self.layer3 = self._make_layer(block,
                                       256,
                                       layers[2],
                                       stride=strides[2],
                                       dilation=dilations[2],
                                       BatchNorm=BatchNorm)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)

        self.stem = [self.conv1, self.bn1]
        self.stages = [self.layer1, self.layer2, self.layer3]

        self._init_weight()
        self.freeze(freeze_at)

    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    stride=1,
                    dilation=1,
                    BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes,
                          planes * block.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, max(dilation // 2, 1),
                  downsample, BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      dilation=dilation,
                      BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        xs = []

        x = self.layer1(x)
        xs.append(x)  # 4X
        x = self.layer2(x)
        xs.append(x)  # 8X
        x = self.layer3(x)
        xs.append(x)  # 16X
        # Following STMVOS, we drop stage 5.
        xs.append(x)  # 16X

        return xs

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def freeze(self, freeze_at):
        if freeze_at >= 1:
            for m in self.stem:
                freeze_params(m)

        for idx, stage in enumerate(self.stages, start=2):
            if freeze_at >= idx:
                freeze_params(stage)


class Decode_Block(nn.Module):
    def __init__(self, in_chans, out_chans, kernel_size, stride, padding=0):
        super().__init__()
        self.linear = nn.ConvTranspose2d(in_chans, out_chans, kernel_size, stride, padding=padding, bias=False)
        self.linear2 = nn.Conv2d(out_chans, out_chans, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.linear(x)
        out = self.linear2(x)
        return x, out


class ResNet_TopDown(ResNet):
    def __init__(self, block, layers, output_stride, BatchNorm, freeze_at=0, use_mask=False):
        super().__init__(block, layers, output_stride, BatchNorm, freeze_at)
        dims = [64, 256, 512, 1024]
        self.downsample_layers = []
        self.downsample_layers.append(
            nn.Sequential(
                self.conv1,
                self.bn1,
                self.relu,
                self.maxpool,
            ))
        self.downsample_layers.append(self.layer1)
        self.downsample_layers.append(self.layer2)
        self.downsample_layers.append(self.layer3)

        self.decoders = nn.ModuleList()
        self.decoders.append(
            nn.Sequential(
                nn.ConvTranspose2d(dims[0], dims[0], 3, 2, 1), # maxpool
                Decode_Block(dims[0], 3, kernel_size=7, stride=2, padding=3), # conv1
            ))
        for i in range(3):
            self.decoders.append(
                Decode_Block(
                    dims[i + 1], dims[i], kernel_size=3, stride=self.strides[i], padding=1,
                ))
        self.prompt = torch.nn.parameter.Parameter(torch.randn(dims[-1]), requires_grad=True)
        self.top_down_transform = torch.nn.parameter.Parameter(torch.eye(dims[-1]), requires_grad=True)

        self.use_mask = use_mask

    def forward_features(self, x, td=None):
        in_var = []
        out_var = []
        for i in range(4):
            in_var.append(x)
            if td is not None:
                x = x + td[i]
            x = self.downsample_layers[i](x)
            out_var.append(x)
        return x, in_var, out_var

    def feedback(self, x):
        td = []
        for depth in range(len(self.decoders) - 1, -1, -1):
            x, out = self.decoders[depth](x)
            td = [out] + td
        return td

    def forward(self, x, mask=None):
        input = x
        x, _, out_var = self.forward_features(input)

        if self.use_mask:
            mask = mask.detach().float()
            mask = transformF.resize(mask, x.shape[2:], InterpolationMode.BILINEAR)
        else:
            cos_sim = (F.normalize(x, dim=1) * F.normalize(
                self.prompt[None, ..., None, None], dim=1)).sum(dim=1, keepdim=True)  # B, N, 1
            mask = cos_sim.clamp(0, 1)
        x = x * mask
        x = (x.permute(0, 2, 3, 1) @ self.top_down_transform).permute(0, 3, 1, 2)
        td = self.feedback(x)

        x, in_var, out_var = self.forward_features(input, td)

        var_loss = self.var_loss(in_var, out_var, x)

        return out_var[1:] + [out_var[-1]], var_loss

    def var_loss(self, in_var, out_var, x):
        recon_loss = []
        for depth in range(len(self.decoders) - 1, -1, -1):
            recon, out = self.decoders[depth](out_var[depth].detach())
            target = in_var[depth].detach()
            recon_loss.append(F.mse_loss(recon, target))

        return sum(recon_loss)


def ResNet50(output_stride, BatchNorm, freeze_at=0):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3],
                   output_stride,
                   BatchNorm,
                   freeze_at=freeze_at)
    return model


def ResNet50_TopDown(output_stride, BatchNorm, freeze_at=0, use_mask=False):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_TopDown(Bottleneck, [3, 4, 6, 3],
                   output_stride,
                   BatchNorm,
                   freeze_at=freeze_at,
                   use_mask=use_mask,
                )
    return model


def ResNet101(output_stride, BatchNorm, freeze_at=0):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3],
                   output_stride,
                   BatchNorm,
                   freeze_at=freeze_at)
    return model


if __name__ == "__main__":
    import torch
    model = ResNet101(BatchNorm=nn.BatchNorm2d, output_stride=8)
    input = torch.rand(1, 3, 512, 512)
    output, low_level_feat = model(input)
    print(output.size())
    print(low_level_feat.size())
