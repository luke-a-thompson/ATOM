"""Train with ATOM init/FFN fixes patched in, without touching library code.

FIX_INIT=1   keep the MLP's zero-initialised final layer (ATOM._initialise_weights overwrites it)
FIX_SWIGLU=1 give each SwiGLU position its own weights (MLP reuses one module twice)
"""
import copy, os, sys
import torch.nn as nn
import atom.training  # noqa: F401  (import order avoids a circular import)
import atom.atom.atom_model as am
import atom.atom.mlps as mlps

if os.environ.get("FIX_SWIGLU") == "1":
    orig_mlp_init = mlps.MLP.__init__
    def mlp_init(self, in_dim, hidden_dim, out_dim, hidden_layers, activation, dropout_p):
        orig_mlp_init(self, in_dim, hidden_dim, out_dim, hidden_layers, activation, dropout_p)
        seen = set()
        for i, l in enumerate(self.layers):
            if l is activation:
                if id(l) in seen:
                    self.layers[i] = copy.deepcopy(activation)
                seen.add(id(l))
    mlps.MLP.__init__ = mlp_init
    am.MLP.__init__ = mlp_init

if os.environ.get("FIX_INIT") == "1":
    orig = am.ATOM._initialise_weights
    @staticmethod
    def init(model):
        orig(model)
        for m in model.modules():
            if isinstance(m, mlps.MLP):
                nn.init.zeros_(m.layers[-1].weight); nn.init.zeros_(m.layers[-1].bias)
    am.ATOM._initialise_weights = init

from atom.train import main
sys.argv = ["train"] + sys.argv[1:]
main()
