import torch
import torch.nn as nn
from typing import Any
from pathlib import Path


def clean_state_dict_prefixes(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """
    Remove the '_orig_mod.' prefix from the state_dict keys that is added by torch.compile.
    """
    new_state_dict: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_key = k[len("_orig_mod.") :]
        else:
            new_key = k
        new_state_dict[new_key] = v

    return new_state_dict


def analyze_model_architecture(model_path: str, max_depth: int | None = None) -> dict[str, Any]:
    """
    Analyze a PyTorch model file and return its architecture information.

    Args:
        model_path: Path to the .pt file
        max_depth: Maximum depth to traverse (None for unlimited)

    Returns:
        Dictionary containing model architecture information
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Load the model
    checkpoint = torch.load(model_path, map_location="cpu")

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Clean the state_dict prefixes
    state_dict = clean_state_dict_prefixes(state_dict)

    architecture: dict[str, Any] = {
        "model_path": model_path,
        "total_parameters": 0,
        "modules": {},
        "parameter_groups": {},
    }

    def analyze_module(name: str, module: nn.Module, depth: int = 0) -> dict[str, Any]:
        """Recursively analyze a module and its children."""
        if max_depth is not None and depth > max_depth:
            return {"type": str(type(module).__name__), "truncated": True}

        module_info: dict[str, Any] = {
            "type": type(module).__name__,
            "parameters": {},
            "children": {},
        }

        # Count parameters for this module
        param_count = sum(p.numel() for p in module.parameters())
        module_info["parameter_count"] = param_count
        architecture["total_parameters"] += param_count

        # Analyze children
        for child_name, child_module in module.named_children():
            module_info["children"][child_name] = analyze_module(f"{name}.{child_name}" if name else child_name, child_module, depth + 1)

        return module_info

    # Group parameters by module
    for param_name, param_tensor in state_dict.items():
        module_name = param_name.split(".")[0] if "." in param_name else param_name
        if module_name not in architecture["parameter_groups"]:
            architecture["parameter_groups"][module_name] = {}

        architecture["parameter_groups"][module_name][param_name] = {
            "shape": list(param_tensor.shape),
            "dtype": str(param_tensor.dtype),
            "requires_grad": param_tensor.requires_grad if hasattr(param_tensor, "requires_grad") else None,
        }

    return architecture


def print_model_architecture(
    model_path: str,
    max_depth: int | None = None,
    show_parameters: bool = True,
    show_shapes: bool = True,
) -> None:
    """
    Load a PyTorch model file and print its architecture in a pretty format.

    Args:
        model_path: Path to the .pt file
        max_depth: Maximum depth to traverse (None for unlimited)
        show_parameters: Whether to show parameter information
        show_shapes: Whether to show tensor shapes
    """
    try:
        # Load the model to get the actual model object for hierarchical display
        checkpoint = torch.load(model_path, map_location="cpu")

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
                model = checkpoint.get("model", None)
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
                model = checkpoint["model"]
            else:
                state_dict = checkpoint
                model = None
        else:
            state_dict = checkpoint
            model = None

        print(f"\n{'=' * 60}")
        print("MODEL ARCHITECTURE ANALYSIS")
        print(f"{'=' * 60}")
        print(f"Model file: {model_path}")
        print(f"{'=' * 60}\n")

        # If we have the actual model object, print it hierarchically
        if model is not None:
            print("MODEL HIERARCHY:")
            print("-" * 40)
            print_model_tree(model, max_depth)
        else:
            # Fallback to parameter-based analysis
            arch = analyze_model_architecture(model_path, max_depth)
            print(f"Total parameters: {arch['total_parameters']:,}")

            if show_parameters:
                print("\nPARAMETER GROUPS:")
                print("-" * 40)
                for module_name, params in arch["parameter_groups"].items():
                    print(f"\n{module_name}:")
                    for param_name, param_info in params.items():
                        shape_str = f"shape={param_info['shape']}" if show_shapes else ""
                        dtype_str = f"dtype={param_info['dtype']}" if show_shapes else ""
                        info_parts = [shape_str, dtype_str] if show_shapes else []
                        info_str = f" ({', '.join(filter(None, info_parts))})" if info_parts else ""
                        print(f"  {param_name}{info_str}")

            print(f"\n{'=' * 60}")
            print("MODULE HIERARCHY (from parameters):")
            print("-" * 40)
            for module_name in arch["parameter_groups"].keys():
                print(f"├── {module_name}")

        print(f"\n{'=' * 60}")

    except Exception as e:
        print(f"Error analyzing model: {e}")


def print_model_tree(model: nn.Module, max_depth: int | None = None, prefix: str = "", depth: int = 0) -> None:
    """
    Recursively print the model hierarchy in a tree structure.

    Args:
        model: The PyTorch model to print
        max_depth: Maximum depth to traverse
        prefix: Current prefix for indentation
        depth: Current depth level
    """
    if max_depth is not None and depth > max_depth:
        return

    # Get module name and type
    module_name = model.__class__.__name__

    # Get parameter count for this module
    param_count = sum(p.numel() for p in model.parameters(recurse=False))
    param_str = f" ({param_count:,} params)" if param_count > 0 else ""

    # Print current module
    print(f"{prefix}{module_name}{param_str}")

    # Print children
    for name, child in model.named_children():
        child_prefix = f"{prefix}  " if prefix else "  "
        print(f"{child_prefix}({name}): ", end="")
        print_model_tree(child, max_depth, "", depth + 1)


def compare_model_architectures(model_path1: str, model_path2: str, max_depth: int | None = None) -> None:
    """
    Compare two PyTorch model files and highlight differences.

    Args:
        model_path1: Path to the first .pt file
        model_path2: Path to the second .pt file
        max_depth: Maximum depth to traverse (None for unlimited)
    """
    try:
        arch1 = analyze_model_architecture(model_path1, max_depth)
        arch2 = analyze_model_architecture(model_path2, max_depth)

        print(f"\n{'=' * 80}")
        print("MODEL COMPARISON")
        print(f"{'=' * 80}")
        print(f"Model 1: {arch1['model_path']}")
        print(f"Model 2: {arch2['model_path']}")
        print(f"{'=' * 80}\n")

        # Compare parameter counts
        print("PARAMETER COUNT COMPARISON:")
        print("-" * 40)
        print(f"Model 1 total parameters: {arch1['total_parameters']:,}")
        print(f"Model 2 total parameters: {arch2['total_parameters']:,}")
        diff = arch2["total_parameters"] - arch1["total_parameters"]
        print(f"Difference: {diff:+,}")

        # Compare parameter groups
        print("\nPARAMETER GROUP COMPARISON:")
        print("-" * 40)

        all_modules = set(arch1["parameter_groups"].keys()) | set(arch2["parameter_groups"].keys())

        for module in sorted(all_modules):
            params1 = arch1["parameter_groups"].get(module, {})
            params2 = arch2["parameter_groups"].get(module, {})

            count1 = len(params1)
            count2 = len(params2)

            if count1 != count2:
                print(f"⚠️  {module}: {count1} vs {count2} parameters")
            else:
                print(f"✓ {module}: {count1} parameters (same)")

        # Show detailed differences in parameter shapes
        print("\nDETAILED PARAMETER DIFFERENCES:")
        print("-" * 40)

        for module in sorted(all_modules):
            params1 = arch1["parameter_groups"].get(module, {})
            params2 = arch2["parameter_groups"].get(module, {})

            all_params = set(params1.keys()) | set(params2.keys())

            for param in sorted(all_params):
                if param not in params1:
                    print(f"➕ {module}.{param}: NEW in Model 2")
                elif param not in params2:
                    print(f"➖ {module}.{param}: REMOVED from Model 2")
                else:
                    shape1 = params1[param]["shape"]
                    shape2 = params2[param]["shape"]
                    if shape1 != shape2:
                        print(f"🔄 {module}.{param}: {shape1} → {shape2}")

        print(f"\n{'=' * 80}")

    except Exception as e:
        print(f"Error comparing models: {e}")


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python model_arch_viz.py <model_path> [model_path2]")
        print("  Single model: python model_arch_viz.py model.pt")
        print("  Compare models: python model_arch_viz.py model1.pt model2.pt")
        sys.exit(1)

    model_path = sys.argv[1]

    if len(sys.argv) == 2:
        print_model_architecture(model_path)
    elif len(sys.argv) == 3:
        model_path2 = sys.argv[2]
        compare_model_architectures(model_path, model_path2)
    else:
        print("Too many arguments. Use: python model_arch_viz.py <model_path> [model_path2]")
