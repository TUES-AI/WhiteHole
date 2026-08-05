from pldm.adaptation.adapters import AdapterDataConfig, build_dataloaders


def build_adapter_dataloaders(
    config: AdapterDataConfig, source_config_path, val_batches
):
    """Build source/target transition loaders for adapter training."""

    return build_dataloaders(source_config_path, config, val_batches)
