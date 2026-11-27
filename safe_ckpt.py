"""Load the VMamba ImageNet checkpoint without running arbitrary pickled code.

The released vmamba_small_e238_ema.pth also pickles a yacs CfgNode (the training
config), which PyTorch >= 2.6 refuses in weights_only mode. Instead of falling
back to an unrestricted torch.load, this unpickler allows only tensor rebuilding,
plain containers and the config class (loaded as a plain dict). Anything else
raises an error. Written for this project.
"""
import collections
import pickle
import types

import torch

class _Cfg(dict):
    """Stand-in for yacs CfgNode: a plain dict that can also hold attributes."""


_ALLOWED = {
    ("__builtin__", "set"): set,
    ("builtins", "set"): set,
    ("collections", "OrderedDict"): collections.OrderedDict,
    ("torch._utils", "_rebuild_tensor_v2"): torch._utils._rebuild_tensor_v2,
    ("torch._utils", "_rebuild_parameter"): torch._utils._rebuild_parameter,
    ("torch", "Size"): torch.Size,
    ("yacs.config", "CfgNode"): _Cfg,
}


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in _ALLOWED:
            return _ALLOWED[(module, name)]
        if module == "torch" and name.endswith("Storage"):
            return getattr(torch, name)
        raise pickle.UnpicklingError(f"blocked global {module}.{name}")


_mod = types.ModuleType("safe_pickle")
_mod.Unpickler = _Unpickler
_mod.load = pickle.load
_mod.__name__ = "safe_pickle"


def load(path):
    return torch.load(path, map_location="cpu", pickle_module=_mod, weights_only=False)


def load_vmamba_into(model, path):
    """Same mapping as the official VMUNet.load_from(): encoder keys as they are,
    then encoder stage i copied into decoder stage 3-i."""
    pre = load(path)["model"]
    md = model.vmunet.state_dict()
    enc = {k: v for k, v in pre.items() if k in md}
    md.update(enc)
    remap = {"layers.0": "layers_up.3", "layers.1": "layers_up.2", "layers.2": "layers_up.1", "layers.3": "layers_up.0"}
    dec = {}
    for k, v in pre.items():
        for old, new in remap.items():
            if old in k:
                nk = k.replace(old, new)
                if nk in md and md[nk].shape == v.shape:
                    dec[nk] = v
                break
    md.update(dec)
    model.vmunet.load_state_dict(md)
    print(f"VMamba-S weights: {len(enc)} encoder and {len(dec)} decoder tensors loaded of {len(md)}")
    return model
