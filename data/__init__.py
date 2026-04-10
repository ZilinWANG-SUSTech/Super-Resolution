from .datamodule import SRDataModule, GuidedSRDataModule


def build_datamodule(config_dict: dict):
    module_type = config_dict.pop('type', 'SRDataModule')

    if module_type == "SRDataModule":
        return SRDataModule(**config_dict)
    elif module_type == "GuidedSRDataModule":
        return GuidedSRDataModule(**config_dict)
    else:
        raise ValueError(f"Unknown datamodule type: {module_type}")
