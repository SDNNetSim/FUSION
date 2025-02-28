import unittest
import json
import os
from unittest.mock import mock_open, patch

from data_scripts.generate_data import create_pt, create_bw_info


class TestGenerateData(unittest.TestCase):
    """
    Tests functions in generate_data.py script.
    """

    @patch('builtins.open', new_callable=mock_open, read_data=json.dumps({
        "nodes": {
            "A": {"type": "CDC"},
            "B": {"type": "CDC"},
            "C": {"type": "CDC"}
        },
        "links": {
            "1": {
                "fiber": {
                    "attenuation": 4.605111673958094e-05,
                    "non_linearity": 0.0013,
                    "dispersion": 2.0393053374841523e-26,
                    "num_cores": 4,
                    "fiber_type": 0,
                    "bending_radius": 0.05,
                    "mode_coupling_co": 0.0004,
                    "propagation_const": 4000000.0,
                    "core_pitch": 4e-05
                },
                "length": 100,
                "source": "A",
                "destination": "B",
                "span_length": 100
            },
            "2": {
                "fiber": {
                    "attenuation": 4.605111673958094e-05,
                    "non_linearity": 0.0013,
                    "dispersion": 2.0393053374841523e-26,
                    "num_cores": 4,
                    "fiber_type": 0,
                    "bending_radius": 0.05,
                    "mode_coupling_co": 0.0004,
                    "propagation_const": 4000000.0,
                    "core_pitch": 4e-05
                },
                "length": 150,
                "source": "B",
                "destination": "C",
                "span_length": 100
            }
        }
    }))
    def test_create_pt(self, mock_file):
        """
        Tests creating physical topology.
        """
        cores_per_link = 4
        net_spec_dict = {('A', 'B'): 100, ('B', 'C'): 150}

        topology_dict = create_pt(cores_per_link, net_spec_dict)

        self.assertIn('nodes', topology_dict)
        self.assertIn('links', topology_dict)
        self.assertEqual(len(topology_dict['nodes']), 3)
        self.assertEqual(len(topology_dict['links']), 2)

        for _, props in topology_dict['nodes'].items():
            self.assertEqual(props, {'type': 'CDC'})

        for link_num, props in topology_dict['links'].items():
            self.assertIn('fiber', props)
            self.assertIn('length', props)
            self.assertIn('source', props)
            self.assertIn('destination', props)
            self.assertIn('span_length', props)

        exp_top_dict = json.load(mock_file())
        check_top_dict = {'nodes': exp_top_dict['nodes'], 'links': {}}
        for link_num, data_dict in exp_top_dict['links'].items():
            check_top_dict['links'][int(link_num)] = data_dict
        self.assertEqual(topology_dict, check_top_dict)

    def test_create_bw_info_example_mod_a(self):
        """
        Tests creating Yue's bandwidth assumptions.
        """
        mod_assumption = 'example_mod_a'
        input_mod_format = os.path.join('tests', 'fixtures', 'test_mod_formats.json')
        with open(input_mod_format, 'r', encoding='utf-8') as mod_format_obj:
            expected_bw_mod_dict = json.load(mod_format_obj)
        bw_mod_dict = create_bw_info(mod_assumption, input_mod_format)
        self.assertEqual(bw_mod_dict, expected_bw_mod_dict[mod_assumption])

    def test_create_bw_info_example_mod_b(self):
        """
        Tests creating Arash's bandwidth assumptions.
        """
        mod_assumption = 'example_mod_b'
        input_mod_format = os.path.join('tests', 'fixtures', 'test_mod_formats.json')
        with open(input_mod_format, 'r', encoding='utf-8') as mod_format_obj:
            expected_bw_mod_dict = json.load(mod_format_obj)
        bw_mod_dict = create_bw_info(mod_assumption, input_mod_format)
        self.assertEqual(bw_mod_dict, expected_bw_mod_dict[mod_assumption])

    def test_create_bw_info_invalid(self):
        """
        Tests an invalid bandwidth assumption.
        """
        mod_assumption = 'invalid'
        mod_assumption_path = os.path.join('tests', 'fixtures', 'test_mod_formats.json')
        with self.assertRaises(NotImplementedError):
            create_bw_info(mod_assumption, mod_assumption_path)


if __name__ == '__main__':
    unittest.main()
