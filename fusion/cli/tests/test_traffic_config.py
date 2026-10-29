"""Traffic options through the real INI and CLI loaders."""

from pathlib import Path

import pytest

from fusion.cli.config_setup import load_config
from fusion.cli.main_parser import build_parser


@pytest.fixture
def ini(tmp_path):
    template = Path(__file__).resolve().parents[2] / "configs/templates/static_design.ini"
    path = tmp_path / "design.ini"
    path.write_text(template.read_text())
    return path


def test_traffic_cli_overrides_absent_ini_options(ini):
    ini.write_text("\n".join(line for line in ini.read_text().splitlines() if not line.startswith(("traffic_", "request_sort"))))
    args = build_parser().parse_args(
        [
            "run_sim",
            "--config_path",
            str(ini),
            "--run_id",
            "traffic-test",
            "--traffic_type",
            "semi_static",
            "--request_sort",
            "bandwidth",
        ]
    )
    config = load_config(str(ini), vars(args))["s1"]
    assert config["traffic_type"] == "semi_static"
    assert config["request_sort"] == "bandwidth"


def test_matrix_paths_relative_to_config_and_thread_overrides(ini):
    ini.write_text(
        ini.read_text().replace("traffic_matrix_file =", "traffic_matrix_file = demands.csv") + "\n[s2]\ntraffic_type = semi_static\n"
    )
    config = load_config(str(ini))
    assert config["s1"]["traffic_matrix_file"] == str(ini.parent / "demands.csv")
    assert config["s1"]["traffic_type"] == "static"
    assert config["s2"]["traffic_type"] == "semi_static"


@pytest.mark.parametrize(
    "old,new",
    [
        ("traffic_type = static", "traffic_type = typo"),
        ("traffic_matrix_scale = 1.0", "traffic_matrix_scale = nonsense"),
        ("traffic_matrix_scale = 1.0", "traffic_matrix_scale = 0"),
    ],
)
def test_invalid_traffic_config_is_not_silently_ignored(ini, old, new):
    ini.write_text(ini.read_text().replace(old, new))
    assert load_config(str(ini)) == {}
