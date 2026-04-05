from .registry import ENGINE_REGISTRY, NETWORK_REGISTRY, DIFFUSION_REGISTRY


def build_engine(config: dict):
    """
    Builds the Lightning module dynamically from the config using the registry.
    """
    if "target" not in config:
        raise KeyError("Expected key 'target' to instantiate from config.")
        
    engine_name = config["target"]
    
    # Look up the class directly from the global dictionary!
    engine_cls = ENGINE_REGISTRY.get(engine_name)
    
    if engine_cls is None:
        raise KeyError(f"Engine '{engine_name}' not found in ENGINE_REGISTRY. "
                       f"Make sure you have imported the module.")
                       
    params = config.get("params", {})
    return engine_cls(**params)

def build_engine_cls(config: dict):
    """
    Builds the Lightning module dynamically from the config using the registry.
    """
    if "target" not in config:
        raise KeyError("Expected key 'target' to instantiate from config.")
        
    engine_name = config["target"]
    
    # Look up the class directly from the global dictionary!
    engine_cls = ENGINE_REGISTRY.get(engine_name)
    
    if engine_cls is None:
        raise KeyError(f"Engine '{engine_name}' not found in ENGINE_REGISTRY. "
                       f"Make sure you have imported the module.")
                       
    return engine_cls


def build_network(config: dict):
    """
    Builds the Lightning module dynamically from the config using the registry.
    """
    if "target" not in config:
        raise KeyError("Expected key 'target' to instantiate from config.")
        
    network_name = config["target"]
    
    # Look up the class directly from the global dictionary!
    network_cls = NETWORK_REGISTRY.get(network_name)
    
    if network_cls is None:
        raise KeyError(f"Network '{network_name}' not found in NETWORK_REGISTRY. "
                       f"Make sure you have imported the module."
                       f"NETWORK_REGISTRY has {NETWORK_REGISTRY._module_dict}"
                       )
                       
    params = config.get("params", {})
    return network_cls(**params)

def build_diffusion(config: dict):
    if "target" not in config:
        raise KeyError("Expected key 'target' to instantiate from config.")
        
    diffusion_name = config["target"]
    
    diffusion_builder = DIFFUSION_REGISTRY.get(diffusion_name)
    
    if diffusion_builder is None:
        raise KeyError(
            f"Diffusion method '{diffusion_name}' not found in DIFFUSION_REGISTRY. "
            f"Make sure you have imported the module where it is defined. "
            f"Current DIFFUSION_REGISTRY has: {list(DIFFUSION_REGISTRY._module_dict.keys())}"
        )
    
    if not callable(diffusion_builder):
        raise TypeError(f"The registered target '{diffusion_name}' is not callable.")
                       
    params = config.get("params", {})
    
    return diffusion_builder(**params)