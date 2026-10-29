"""TopologyBench options through the public CLI/INI configuration path."""

from pathlib import Path

from fusion.cli.config_setup import load_config
from fusion.cli.main_parser import build_parser


def template():
    return (Path(__file__).resolve().parents[2] / "configs/templates/static_design.ini").read_text()


def test_topology_options_and_paths_follow_thread_overrides(tmp_path):
    path = tmp_path / "topology.ini"
    path.write_text(
        template().replace("[topology_settings]", "[topology_settings]\ntopology_file = real.zip\ntopology_name = TOP_1_ABILENE")
        + "\n[s2]\ntopology_file = second.xlsx\ntopology_name = second\n"
    )
    config = load_config(str(path))
    assert config["s1"]["topology_file"] == str(tmp_path / "real.zip")
    assert config["s1"]["topology_name"] == "TOP_1_ABILENE"
    assert config["s2"]["topology_file"] == str(tmp_path / "second.xlsx")
    assert config["s2"]["topology_name"] == "second"


def test_topology_cli_options_override_old_config(tmp_path):
    path = tmp_path / "old.ini"
    path.write_text(template())
    args = build_parser().parse_args(
        [
            "run_sim",
            "--config_path",
            str(path),
            "--run_id",
            "test-topology",
            "--topology_file",
            "real.zip",
            "--topology_name",
            "TOP_1_ABILENE",
        ]
    )
    config = load_config(str(path), vars(args))["s1"]
    assert config["topology_file"] == str(tmp_path / "real.zip")
    assert config["topology_name"] == "TOP_1_ABILENE"


def test_selector_without_file_is_rejected(tmp_path):
    path = tmp_path / "invalid.ini"
    path.write_text(template().replace("[topology_settings]", "[topology_settings]\ntopology_name = anything"))
    assert load_config(str(path)) == {}
