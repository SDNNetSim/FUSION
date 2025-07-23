def add_gui_args(parser):
    parser.add_argument("--config_path", type=str, required=True, help="Path to INI config file.")
    parser.add_argument("--run_id", type=str, required=True, help="Run identifier.")
    # Add more GUI-specific CLI args here if needed (e.g., --debug, --theme)
