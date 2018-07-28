from __future__ import absolute_import
from collections import OrderedDict

from torch.autograd import Variable

from ..utils import to_torch
import pdb

def extract_cnn_feature(model, inputs, modules=None):
    model.eval()
    inputs = to_torch(inputs)
    inputs = Variable(inputs.cuda(), volatile=True)
    if modules is None:
        output_maps, outputs = model(inputs)
        outputs = outputs.data.cpu()
        output_maps = output_maps.data.cpu()
        return outputs, output_maps
    # Register forward hook for each module
    outputs = OrderedDict()
    handles = []
    for m in modules:
        outputs[id(m)] = None
        def func(m, i, o): outputs[id(m)] = o.data.cpu()
        handles.append(m.register_forward_hook(func))
    model(inputs)
    for h in handles:
        h.remove()
    return list(outputs.values())
