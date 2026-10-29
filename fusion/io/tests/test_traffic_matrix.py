"""Traffic input formats and validation tests."""

import json
from unittest.mock import MagicMock, patch

import pytest

from fusion.io.traffic_matrix import download_traffic_matrix, load_traffic_matrix, prepare_traffic_matrix


@pytest.mark.parametrize(
    "suffix,content,fmt",
    [
        ("csv", "source,destination,bandwidth,count\nA,B,100,2\n", "auto"),
        ("json", '[{"source":"A","destination":"B","bandwidth":100,"count":2}]', "auto"),
        ("json", '{"demands":[{"source":"A","destination":"B","bandwidth":100,"count":2}]}', "auto"),
        ("csv", ",A,B\nA,0,100\nB,100,0\n", "matrix_csv"),
    ],
)
def test_input_formats(tmp_path, suffix, content, fmt):
    path = tmp_path / f"traffic.{suffix}"
    path.write_text(content)
    rows = load_traffic_matrix(str(path), fmt, scale=2)
    assert len(rows) == 2
    assert all(row["bandwidth"] == "200" for row in rows)


def test_sndlib_namespace_and_routing_units(tmp_path):
    path = tmp_path / "network.xml"
    path.write_text("""<network xmlns="http://sndlib.zib.de/network"><demands>
    <demand id="d1"><source>A</source><target>B</target><routingUnit>10</routingUnit>
    <demandValue>20</demandValue></demand></demands></network>""")
    assert load_traffic_matrix(str(path)) == [{"source": "A", "destination": "B", "bandwidth": "200"}]


@pytest.mark.parametrize(
    "record,match",
    [
        ({"source": "A", "destination": "A", "bandwidth": 100}, "must differ"),
        ({"destination": "B", "bandwidth": 100}, "missing source"),
        ({"source": "A", "destination": "B", "bandwidth": -1}, "positive"),
        ({"source": "A", "destination": "B", "bandwidth": "NaN"}, "finite"),
        ({"source": "A", "destination": "B", "bandwidth": 100, "count": 0}, "positive"),
        ({"source": "A", "destination": "B", "bandwidth": 100, "count": 1.5}, "integer"),
        ({"source": "A", "destination": "B", "bandwidth": 100, "arrival": "inf"}, "finite"),
    ],
)
def test_invalid_records(tmp_path, record, match):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([record]))
    with pytest.raises(ValueError, match=match):
        load_traffic_matrix(str(path))


@pytest.mark.parametrize(
    "content",
    [
        ",A,B\nA,0,100\n",
        ",A,A\nA,0,100\nA,100,0\n",
        ",A,B\nA,1,100\nB,100,0\n",
        ",A,B\nA,0,-1\nB,100,0\n",
    ],
)
def test_malformed_square_matrices(tmp_path, content):
    path = tmp_path / "bad.csv"
    path.write_text(content)
    with pytest.raises(ValueError):
        load_traffic_matrix(str(path), "matrix_csv")


def test_mixed_arrival_times_rejected(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("source,destination,bandwidth,arrival\nA,B,100,2\nA,B,100,\n")
    with pytest.raises(ValueError, match="all demands"):
        load_traffic_matrix(str(path))


def test_node_mapping_and_bandwidth_validation(tmp_path):
    path = tmp_path / "input.csv"
    path.write_text("source,destination,bandwidth\nParis,London,100\n")
    mapping = tmp_path / "nodes.json"
    mapping.write_text('{"Paris":"1","London":"2"}')
    props = {
        "traffic_matrix_file": str(path),
        "traffic_node_map_file": str(mapping),
        "topology_info": {"nodes": {"1": {}, "2": {}}},
        "mod_per_bw": {"100": {}},
    }
    prepare_traffic_matrix(props)
    assert props["num_requests"] == 1
    assert props["traffic_demands"][0]["source"] == "1"
    # Prepared inputs remain reproducible even if the source file later changes.
    path.write_text("invalid")
    prepare_traffic_matrix(props)
    assert props["num_requests"] == 1


@pytest.mark.parametrize(
    "nodes,bands,match",
    [
        ({"A": {}}, {"100": {}}, "absent"),
        ({"A": {}, "B": {}}, {"400": {}}, "not supported"),
    ],
)
def test_incompatible_input_rejected(tmp_path, nodes, bands, match):
    path = tmp_path / "input.csv"
    path.write_text("source,destination,bandwidth\nA,B,100\n")
    with pytest.raises(ValueError, match=match):
        prepare_traffic_matrix({"traffic_matrix_file": str(path), "topology_info": {"nodes": nodes}, "mod_per_bw": bands})


def test_public_download_and_no_overwrite(tmp_path):
    response = MagicMock()
    response.__enter__.return_value = response
    response.geturl.return_value = "https://example.org/traffic.csv"
    response.read.return_value = b"source,destination,bandwidth\nA,B,100\n"
    target = tmp_path / "traffic.csv"
    with patch("fusion.io.traffic_matrix.urllib.request.urlopen", return_value=response):
        download_traffic_matrix(response.geturl(), str(target))
        assert len(load_traffic_matrix(str(target))) == 1
        with pytest.raises(FileExistsError):
            download_traffic_matrix(response.geturl(), str(target))
    with pytest.raises(ValueError, match="HTTPS"):
        download_traffic_matrix("file:///private", str(target))


def test_xml_dtd_is_rejected(tmp_path):
    path = tmp_path / "bad.xml"
    path.write_text('<!DOCTYPE network [<!ENTITY x "test">]><network/>')
    with pytest.raises(ValueError, match="DTD"):
        load_traffic_matrix(str(path))


def test_excel_matrix_preserves_fractional_values_and_ignores_padding(tmp_path):
    from openpyxl import Workbook

    path = tmp_path / "traffic.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Traffic"
    sheet.append([None, "A", "B"])
    sheet.append(["A", None, 44.5])
    sheet.append(["B", 44.5, None])
    sheet.cell(15, 3).number_format = "0.0"
    workbook.save(path)
    rows = load_traffic_matrix(str(path), sheet="Traffic", direction="undirected")
    assert rows == [{"source": "A", "destination": "B", "bandwidth": "44.5"}]
    assert len(load_traffic_matrix(str(path), sheet="Traffic")) == 2
    with pytest.raises(ValueError, match="traffic_matrix_sheet"):
        load_traffic_matrix(str(path))
    props = {
        "traffic_matrix_file": str(path),
        "traffic_matrix_sheet": "Traffic",
        "traffic_matrix_direction": "undirected",
        "topology_info": {"nodes": {"A": {}, "B": {}}},
        "mod_per_bw": {"44.5": {}},
    }
    prepare_traffic_matrix(props)
    assert props["traffic_demands"][0]["bandwidth"] == "44.5"
    assert props["traffic_matrix_summary"]["offered_bandwidth"] == 44.5
    assert props["traffic_matrix_summary"]["simulated_bandwidth"] == 44.5


def test_undirected_matrix_requires_symmetry(tmp_path):
    path = tmp_path / "traffic.csv"
    path.write_text(",A,B\nA,0,44.5\nB,50,0\n")
    with pytest.raises(ValueError, match="symmetric"):
        load_traffic_matrix(str(path), "matrix_csv", direction="undirected")


def test_exact_bandwidth_requires_exact_modulation_entry(tmp_path):
    path = tmp_path / "traffic.csv"
    path.write_text("source,destination,bandwidth\nA,B,44.5\n")
    with pytest.raises(ValueError, match="44.5.*not supported"):
        prepare_traffic_matrix(
            {"traffic_matrix_file": str(path), "topology_info": {"nodes": {"A": {}, "B": {}}}, "mod_per_bw": {"100": {}}}
        )


def test_no_significant_digits_are_discarded(tmp_path):
    path = tmp_path / "traffic.csv"
    path.write_text("source,destination,bandwidth\nA,B,412.99999999999994\n")
    rows = load_traffic_matrix(str(path))
    assert rows[0]["bandwidth"] == "412.99999999999994"
