# fusion/cli/run_gui.py

from fusion.cli.main_parser import get_gui_args
from fusion.cli.config_setup import ConfigManager

def main():
    args = get_gui_args()
    config = ConfigManager.from_args(args)

    print("🖥️  GUI launch invoked.")
    print(f"✅ Parsed config for run_id: {args.run_id}")
    print("📂 Config summary (s1):")
    print(config.get("s1"))

    try:
        from fusion.gui.runner import launch_gui
        launch_gui(config)
    except ImportError:
        print("🚧 GUI logic not implemented or runner missing.")

if __name__ == "__main__":
    main()
