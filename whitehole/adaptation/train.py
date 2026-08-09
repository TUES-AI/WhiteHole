from whitehole.adaptation.adapters import AdapterTrainConfig, train_adapter


def main(config: AdapterTrainConfig):
    """Train an appearance adapter on reward-free target transitions."""

    return train_adapter(config)


if __name__ == "__main__":
    main(AdapterTrainConfig.parse_from_command_line())
