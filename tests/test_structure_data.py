import unittest
from pathlib import Path
from unittest.mock import mock_open, patch
from fusion.data_scripts.structure_data import assign_link_lengths, create_network
from fusion.cli.setup_helpers import find_project_root


class TestNetworkFunctions(unittest.TestCase):
    """
    Test structure_data.py
    """

    def test_assign_link_lengths(self):
        """
        Test assign link lengths.
        """
        mock_file_content = "1\t2\t10\n2\t3\t20\n"
        node_pairs_dict = {'1': 'A', '2': 'B', '3': 'C'}
        network_fp = Path('dummy', 'path', 'network.txt')

        with patch("pathlib.Path.open", mock_open(read_data=mock_file_content)):
            response_dict = assign_link_lengths(network_fp=network_fp, node_pairs_dict=node_pairs_dict,
                                                constant_weight=False)
            expected_response = {('A', 'B'): 10.0, ('B', 'C'): 20.0}
            self.assertEqual(response_dict, expected_response)

        with patch("pathlib.Path.open", mock_open(read_data=mock_file_content)):
            response_dict = assign_link_lengths(network_fp=network_fp, node_pairs_dict=node_pairs_dict,
                                                constant_weight=True)
            expected_response = {('A', 'B'): 1.0, ('B', 'C'): 1.0}
            self.assertEqual(response_dict, expected_response)

    @patch("fusion.data_scripts.structure_data.assign_link_lengths")
    def test_create_network(self, mock_assign_link_lengths):
        """
        Test create network.
        """
        mock_assign_link_lengths.return_value = {'link_lengths': 'mocked'}

        base_network_fp = Path(find_project_root()) / 'data' / 'raw'

        net_name = 'USNet'
        response = create_network(net_name=net_name, const_weight=False)
        mock_assign_link_lengths.assert_called_with(
            constant_weight=False,
            network_fp=base_network_fp / 'us_network.txt',
            node_pairs_dict={}
        )
        self.assertEqual(({'link_lengths': 'mocked'}, []), response)

        net_name = 'NSFNet'
        response = create_network(net_name=net_name, const_weight=False)
        mock_assign_link_lengths.assert_called_with(
            constant_weight=False,
            network_fp=base_network_fp / 'nsf_network.txt',
            node_pairs_dict={}
        )
        self.assertEqual(({'link_lengths': 'mocked'}, []), response)

        net_name = 'Pan-European'
        response = create_network(net_name=net_name, const_weight=False)
        mock_assign_link_lengths.assert_called_with(
            constant_weight=False,
            network_fp=base_network_fp / 'europe_network.txt',
            node_pairs_dict={}
        )
        self.assertEqual(({'link_lengths': 'mocked'}, []), response)

        with self.assertRaises(NotImplementedError):
            create_network(net_name="UnknownNet")

    def test_create_network_with_base_fp(self):
        """
        Test create network with base_fp specified.
        """
        base_fp = Path(find_project_root()) / 'custom' / 'path'
        with patch("fusion.data_scripts.structure_data.assign_link_lengths") as mock_assign_link_lengths:
            mock_assign_link_lengths.return_value = {'link_lengths': 'mocked'}

            net_name = 'USNet'
            response = create_network(net_name=net_name, base_fp=base_fp, const_weight=False)
            network_fp = base_fp / 'raw' / 'us_network.txt'
            mock_assign_link_lengths.assert_called_with(
                constant_weight=False,
                network_fp=network_fp,
                node_pairs_dict={}
            )
            self.assertEqual(({'link_lengths': 'mocked'}, []), response)


if __name__ == '__main__':
    unittest.main()
