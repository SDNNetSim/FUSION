from fusion.cli.main_parser import get_train_args
from fusion.cli.config_setup import ConfigManager
from fusion.sim.train_pipeline import train_rl_agent
from fusion.sim.ml_pipeline import train_ml_model


def main():
    """
    Controls the run_train script.
    """
    args = get_train_args()
    config = ConfigManager.from_args(args)

    print("✅ CLI args and config parsed successfully.")
    print("🔁 Dispatching to training pipeline...")

    agent_type = config.get_args().agent_type

    if agent_type == "rl":
        train_rl_agent(config)
    elif agent_type == "ml":
        train_ml_model(config)
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")


if __name__ == "__main__":
    main()
