from whitehole.adaptation.adapters import AdapterEvalConfig, evaluate_adapter


def main(config: AdapterEvalConfig):
    """Evaluate a frozen JEPA backbone plus an appearance adapter."""

    return evaluate_adapter(config)


if __name__ == "__main__":
    main(AdapterEvalConfig.parse_from_command_line())
