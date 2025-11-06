import re
import sys
from typing import List, Union

import torch
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision import transforms


class VGGFeatureExtractor(torch.nn.Module):

    layers: List[List[int]]
    """Nest list of indices of each conv layer in VGG."""

    model: torchvision.models.vgg.VGG
    """Reference pretrained model."""

    NODE_PATTERN = r'^(conv|relu)([1-5])(?:_([1-4]))?$'

    def __init__(
        self,
        keys: Union[str, List[str]]
    ) -> None:
        super().__init__()

        if isinstance(keys, str):
            keys = [keys]

        nodes_dict = {}
        self.keys = []

        def parse_layer(ln: str):
            m = re.match(self.NODE_PATTERN, ln)
            if not m:
                raise ValueError('"{}" is an invalid identifier'.format(ln))
            op, block, layer = m.groups()

            is_relu = int(op == 'relu')
            b = int(block) - 1
            node_fmt = 'features.{:d}'

            if layer is None:
                keys = []
                for i, layer in enumerate(self.layers[b]):
                    node = node_fmt.format(layer + is_relu)
                    nodes_dict[node] = '_'.join((ln, str(i + 1)))
                    keys.append(nodes_dict[node])
            else:
                keys = [ln]
                node = node_fmt.format(self.layers[b][int(layer) - 1] + is_relu)
                nodes_dict[node] = ln

            self.keys.append((ln, keys))

        for k in keys:
            parse_layer(k)

        self.fx = create_feature_extractor(self.model, return_nodes=nodes_dict)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    @torch.cuda.amp.autocast(dtype=torch.float32)
    def forward(
        self,
        x,
        detach: bool = False
    ) -> dict:
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        assert len(x.shape) == 4  # NCHW
        x_norm = self.normalize(x)
        feats_dict = self.fx(x_norm)

        for k in feats_dict.keys():
            # feats_dict[k] = feats_dict[k].squeeze(0)
            if detach:
                feats_dict[k] = feats_dict[k].detach()

        out_dict = {}
        for k, subkeys in self.keys:
            out_dict[k] = torch.concat([feats_dict[sk] for sk in subkeys], dim=1)
            if detach:
                out_dict[k] = out_dict[k].detach()

        return out_dict


class VGG16FeatureExtractor(VGGFeatureExtractor):
    layers = [[0, 2], [5, 7], [10, 12, 14], [17, 19, 21], [24, 26, 28]]
    model = torchvision.models.vgg16(weights='DEFAULT').eval()


class VGG19FeatureExtractor(VGGFeatureExtractor):
    layers = [[0, 2], [5, 7], [10, 12, 14, 16], [19, 21, 23, 25], [28, 30, 32, 34]]
    model = torchvision.models.vgg19(weights='DEFAULT').eval()


def test_fx(fx_type, H=224, W=224):
    if fx_type == 'vgg16':
        fx_cls = VGG16FeatureExtractor
    elif fx_type == 'vgg19':
        fx_cls = VGG19FeatureExtractor
    else:
        raise ValueError('Invalid extractor type "{}"'.format(fx_type))

    all_layers = [
        'conv{:d}_{:d}'.format(i+1, j+1)
        for i, lvl in enumerate(fx_cls.layers)
        for j in range(len(lvl))]
    all_layers += ['conv{:d}'.format(i+1) for i in range(len(fx_cls.layers))]

    fx = fx_cls(all_layers).cuda()
    input = torch.rand((1, 3, H, W), device='cuda')
    output = fx(input)

    for k, v in output.items():
        print('Feature: {}, size: {}'.format(k, tuple(v.shape)))


if __name__ == '__main__':
    test_fx(sys.argv[1])
