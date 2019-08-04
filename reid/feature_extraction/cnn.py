from __future__ import absolute_import
from collections import OrderedDict

import torch
from torch.autograd import Variable

from ..utils import to_torch


def extract_cnn_feature(model, inputs, for_eval, modules=None):
    model.eval()
    inputs = to_torch(inputs)
    # inputs = Variable(inputs, volatile=True)
    if modules is None:
        with torch.no_grad():
            outputs = model(inputs, for_eval)[0]
            # outputs = outputs[0].data.cpu()  # outputs contains [x1, x2, x3]
            if isinstance(outputs, list):
                # outputs = torch.cat([x.unsqueeze(1) for x in outputs], dim=1)
                outputs = [x.data.cpu() for x in outputs]
            else:
                outputs = outputs.data.cpu()
            return outputs  

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
