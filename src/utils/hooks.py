import copy
from collections import OrderedDict

import torch
import torch.nn as nn
from typing import Dict, List, Callable, Optional


class HookPoint(nn.Module):
    """
    An identity module that acts as a hook point.
    """
    def forward(self, x):
        return x

class HookManager:
    """
    A context manager to both cache and patch model activations.
    
    Adds and automatically removes hooks.
    """
    def __init__(self, 
                 model: nn.Module, 
                 layers_to_cache: Optional[List[str]] = None, 
                 patching_hooks: Optional[Dict[str, Callable]] = None):
        
        self.model = model
        self.layers_to_cache = layers_to_cache or []
        self.patching_hooks = patching_hooks or {}
        
        self.activations: Dict[str, torch.Tensor] = {}
        self._hook_handles = []

    def _get_layer(self, name: str) -> nn.Module:
        """Find a layer in the model using its string name (e.g., 'blocks.0.mlp')."""
        module = self.model
        for part in name.split('.'):
            module = getattr(module, part)
        return module

    def _create_combined_hook(self, name: str) -> Callable:
        """Factory for a hook that can both patch and cache."""
        patch_fn = self.patching_hooks.get(name)
        do_cache = name in self.layers_to_cache
        
        def combined_hook(module, input, output):
            if patch_fn is not None:
                output = patch_fn(module, input, output)

            if do_cache:
                self.activations[name] = output.detach()
            
            if patch_fn is not None:
                return output
                
        return combined_hook

    def __enter__(self):
        """Register a combined hook for all targeted layers."""
        all_layer_names = set(self.layers_to_cache) | set(self.patching_hooks.keys())
        
        for name in all_layer_names:
            try:
                layer = self._get_layer(name)
                hook = self._create_combined_hook(name)
                handle = layer.register_forward_hook(hook)
                self._hook_handles.append(handle)
            except AttributeError:
                raise AttributeError(f"Layer '{name}' not found in the model.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Automatically remove all hooks on exit."""
        for handle in self._hook_handles:
            handle.remove()

def add_semantic_hook_points(model: nn.Module, layer_names: List[str]) -> nn.Module:
    """
    Modifies a model in-place to add HookPoint modules after specified layers,
    enabling semantic hooking.

    Args:
        model (nn.Module): The model to be modified.
        layer_names (List[str]): A list of string names of the layers to which
                                 hook points should be added.
    """
    for name in layer_names:
        parent_module = model
        name_parts = name.split('.')
        
        for part in name_parts[:-1]:
            parent_module = getattr(parent_module, part)
            
        layer_name = name_parts[-1]
        original_layer = getattr(parent_module, layer_name)
        
        retrofitted_layer = nn.Sequential(original_layer, HookPoint())
        setattr(parent_module, layer_name, retrofitted_layer)
        
    print(f"Added hook points for: {layer_names}")
    return model


class HookedModel:
    """
    A wrapper that automatically retrofits a PyTorch model with hook points
    and provides a high-level API for caching and intervention experiments.
    """
    def __init__(self, model: nn.Module):
        # Create a deep copy to avoid modifying the original model
        self.model = copy.deepcopy(model)
        
        # Retrofit the copied model with hook points
        self._add_all_hook_points()
        
        # Discover the newly added hook points
        self.hook_points = self._find_hook_points()
        print(f"Model hooked. Found {len(self.hook_points)} hook points.")

    def __repr__(self):
        return self.model.__repr__()

    def _add_all_hook_points(self):
        """
        Iterates through the model and adds a HookPoint after every
        operational layer (i.e., non-container modules).
        """
        # First, gather the names of all layers to be retrofitted
        layers_to_retrofit = []
        for name, module in self.model.named_modules():
            # We target "leaf" modules: those that have no children
            is_leaf_module = len(list(module.children())) == 0
            if is_leaf_module:
                layers_to_retrofit.append(name)
        
        for name in layers_to_retrofit:
            parent_module = self.model
            name_parts = name.split('.')
            for part in name_parts[:-1]:
                parent_module = getattr(parent_module, part)
            
            layer_name = name_parts[-1]
            original_layer = getattr(parent_module, layer_name)
            retrofitted_layer = nn.Sequential(OrderedDict([
                ("layer", original_layer), 
                ("hook", HookPoint())
            ]))
            setattr(parent_module, layer_name, retrofitted_layer)
    
    def _find_hook_points(self) -> List[str]:
        """Automatically discover all HookPoint modules in the model."""
        return [
            name for name, module in self.model.named_modules()
            if isinstance(module, HookPoint)
        ]

    def run_with_cache(self, *args, **kwargs):
        """Runs the model and caches activations from all hook points."""
        with HookManager(self.model, layers_to_cache=self.hook_points) as manager:
            output = self.model(*args, **kwargs)
            return output, manager.activations
            
    def run_with_hooks(self, *args, fwd_hooks: Dict[str, Callable], **kwargs):
        """Runs the model with custom patching hooks applied."""
        with HookManager(self.model, patching_hooks=fwd_hooks) as manager:
            output = self.model(*args, **kwargs)
            return output

    def __call__(self, *args, **kwargs):
        """Makes the wrapper callable like the original model."""
        return self.model(*args, **kwargs)



if __name__ == '__main__':
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.block1 = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(16, 32)),
                ('act', nn.ReLU()),
                ('fc2', nn.Linear(32, 24)),
                ('act2', nn.ReLU())
            ]))
            self.block2 = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(24, 16)),
                ('act', nn.ReLU())
            ]))
            self.clf = nn.Linear(16, 3)


        def forward(self, x):
            x = self.block1(x)
            x = self.block2(x)
            x = self.clf(x)
            return x

    # 1. Instantiate the original model
    original_model = Net()
    
    print("--- Initializing HookedModel ---")
    hooked_model = HookedModel(original_model)
    
    print("\n--- Retrofitted Model Structure (e.g., 'fc' layer) ---")
    print(hooked_model)
    # 3. Use `run_with_cache` to see all available hook points
    print("\n--- Using run_with_cache() ---")
    dummy_input = torch.randn(4, 16)
    _, cache = hooked_model.run_with_cache(dummy_input)

    print(f"\nCached activations for layers: {list(cache.keys())}")
    # Note the new names: 'fc.1' is the hook point after the original 'fc.0' Linear layer
    print(f"Shape of 'fc.1' activation: {cache['block1.fc.hook'].shape}")

    # 4. Use `run_with_hooks` for an intervention
    print("\n--- Using run_with_hooks() to ablate the 'act' layer's output ---")
    
    def ablate_hook(module, input, output):
        print("   -> Patching: Zeroing out output of the 'act' layer.")
        return torch.zeros_like(output)

    # We target the hook point *after* the ReLU layer ('act.0')
    patched_output = hooked_model.run_with_hooks(
        dummy_input,
        fwd_hooks={'block1.act': ablate_hook}
    )
    print("Intervention complete.")