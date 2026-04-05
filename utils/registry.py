class Registry:
    """
    A registry to map strings to classes.
    """
    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def get(self, key: str):
        """Get the class from the registry."""
        return self._module_dict.get(key, None)

    def register(self, name: str = None):
        """
        Decorator to register a class.
        """
        def _register(cls):
            # If name is not provided, use the class name
            key = name if name is not None else cls.__name__
            
            if key in self._module_dict:
                raise KeyError(f"An object named '{key}' was already registered in '{self._name}' registry!")
            
            self._module_dict[key] = cls
            return cls
            
        return _register


ENGINE_REGISTRY = Registry('Engine')
NETWORK_REGISTRY = Registry('Network')
LOSS_REGISTRY = Registry('Loss')
DIFFUSION_REGISTRY = Registry('Diffusion')