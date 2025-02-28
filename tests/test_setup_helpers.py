import unittest
from unittest.mock import patch, mock_open
import os
import json
from helper_scripts.setup_helpers import create_input, save_input


class TestSetupHelpers(unittest.TestCase):
    """
    Tests the setup_helpers.py script.
    """

    def setUp(self):
        self.base_fp = '/fake/base/path'
        self.engine_props = {
            'sim_type': 'test_sim',
            'thread_num': 1,
            'network': 'test_network',
            'date': '2024-08-19',
            'sim_start': '00:00',
            'const_link_weight': 10,
            'cores_per_link': 7,
            'is_only_core_node': True,
            'mod_assumption': 'example_mod_a',
            'mod_assumption_path': None,
        }
        self.core_nodes = list()
        self.bw_info_dict = {'bandwidth': 100}
        self.network_dict = {'nodes': [], 'links': []}
        self.pt_info = {'cores': 7, 'specifications': {}}

    @patch('helper_scripts.setup_helpers.create_bw_info')
    @patch('helper_scripts.setup_helpers.create_network')
    @patch('helper_scripts.setup_helpers.create_pt')
    @patch('helper_scripts.setup_helpers.save_input')
    @patch('builtins.open', new_callable=mock_open, read_data=json.dumps({'bandwidth': 100}))
    def test_create_input(self, mock_open_file, mock_save_input, mock_create_pt, mock_create_network,
                          mock_create_bw_info):
        """ Tests create input. """
        # Setup mock return values
        mock_create_bw_info.return_value = self.bw_info_dict
        mock_create_network.return_value = self.network_dict, self.core_nodes
        mock_create_pt.return_value = self.pt_info

        # Call the function
        result = create_input(self.base_fp, self.engine_props)

        # Assertions
        mock_create_bw_info.assert_called_once_with(mod_assumption=self.engine_props['mod_assumption'],
                                                    mod_assumptions_path=self.engine_props['mod_assumption_path'])
        mock_save_input.assert_called_once_with(
            base_fp=self.base_fp,
            properties=self.engine_props,
            file_name=f"bw_info_{self.engine_props['thread_num']}.json",
            data_dict=self.bw_info_dict
        )
        mock_create_network.assert_called_once_with(
            base_fp=self.base_fp,
            const_weight=self.engine_props['const_link_weight'],
            net_name=self.engine_props['network'],
            is_only_core_node=True,
        )
        mock_create_pt.assert_called_once_with(
            cores_per_link=self.engine_props['cores_per_link'],
            net_spec_dict=self.network_dict
        )
        mock_open_file.assert_called_once_with(
            os.path.join(self.base_fp, 'input', self.engine_props['network'], self.engine_props['date'],
                         self.engine_props['sim_start'], f"bw_info_{self.engine_props['thread_num']}.json"),
            'r', encoding='utf-8'
        )
        self.assertEqual(result['mod_per_bw'], {'bandwidth': 100})
        self.assertEqual(result['topology_info'], self.pt_info)

    @patch('helper_scripts.setup_helpers.create_dir')
    @patch('builtins.open', new_callable=mock_open)
    def test_save_input(self, mock_open_file, mock_create_dir):
        """ Tests save input. """
        # Test data
        data_dict = {'key': 'value'}
        file_name = 'test_file.json'

        # Call the function
        save_input(self.base_fp, self.engine_props, file_name, data_dict)

        # Assertions
        mock_create_dir.assert_any_call(os.path.join(self.base_fp, 'input', self.engine_props['network'],
                                                     self.engine_props['date'], self.engine_props['sim_start']))
        mock_create_dir.assert_any_call(os.path.join('data', 'output'))

        mock_open_file.assert_called_once_with(os.path.join(self.base_fp, 'input', self.engine_props['network'],
                                                            self.engine_props['date'], self.engine_props['sim_start'],
                                                            file_name),
                                               'w', encoding='utf-8')

        # Aggregate all write calls into a single string
        written_content = "".join(call.args[0] for call in mock_open_file().write.mock_calls)
        expected_content = json.dumps(data_dict, indent=4)

        self.assertEqual(written_content, expected_content)


if __name__ == '__main__':
    unittest.main()
