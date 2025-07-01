# pylint: disable=protected-access

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
from helper_scripts.spectrum_helpers import SpectrumHelpers


class TestSpectrumHelpers(unittest.TestCase):
    """Unit tests for the SpectrumHelpers class."""

    def setUp(self):
        """Set up the test environment for SpectrumHelpers."""
        self.engine_props = {
            'allocation_method': 'first_fit',
            'guard_slots': 1
        }
        # Initialize spectrum_props as an object with attributes
        self.spectrum_props = SimpleNamespace(
            path_list=[1, 2, 3],
            slots_needed=2,
            is_free=False,
            forced_core=None,
            forced_band=None,
            forced_index=None,
            start_slot=None,
            end_slot=None,
            core_num=None,
            curr_band=None
        )
        self.sdn_props = MagicMock()
        # Initialize cores_matrix as a 2D array (matrix) with 2 cores and 10 slots each
        self.sdn_props.net_spec_dict = {
            (1, 2): {'cores_matrix': {'c': np.zeros((2, 10)), 'l': np.zeros((2, 10)), 's': np.zeros((2, 10))}},
            (2, 1): {'cores_matrix': {'c': np.zeros((2, 10)), 'l': np.zeros((2, 10)), 's': np.zeros((2, 10))}},
            (2, 3): {'cores_matrix': {'c': np.zeros((2, 10)), 'l': np.zeros((2, 10)), 's': np.zeros((2, 10))}},
            (3, 2): {'cores_matrix': {'c': np.zeros((2, 10)), 'l': np.zeros((2, 10)), 's': np.zeros((2, 10))}},
        }
        self.helpers = SpectrumHelpers(self.engine_props, self.sdn_props, self.spectrum_props)

    def test_check_free_spectrum(self):
        """Test the _check_free_spectrum method."""
        # Set necessary variables
        self.helpers.curr_band = 'c'
        self.helpers.core_num = 0
        self.helpers.start_index = 0
        self.helpers.end_index = 5

        result = self.helpers._check_free_spectrum((1, 2), (2, 3))
        self.assertTrue(result)

    def test_check_other_links(self):
        """Test the check_other_links method."""
        # Set necessary variables
        self.helpers.curr_band = 'c'
        self.helpers.core_num = 0
        self.helpers.start_index = 0
        self.helpers.end_index = 5

        self.helpers.check_other_links()
        self.assertTrue(self.spectrum_props.is_free)

    def test_update_spec_props(self):
        """Test the _update_spec_props method."""
        # Set necessary variables
        self.helpers.curr_band = 'c'
        self.helpers.core_num = 0
        self.helpers.start_index = 0
        self.helpers.end_index = 5

        self.helpers._update_spec_props()
        self.assertEqual(self.spectrum_props.start_slot, 0)
        self.assertEqual(self.spectrum_props.end_slot, 6)
        self.assertEqual(self.spectrum_props.core_num, 0)
        self.assertEqual(self.spectrum_props.curr_band, 'c')

    def test_check_super_channels(self):
        """Test checking for available super-channels."""
        open_slots_matrix = [[0, 1, 2, 3, 4], [5, 6, 7, 8]]

        # Set necessary variables
        self.helpers.curr_band = 'c'
        self.helpers.core_num = 0

        # Test when there is a valid allocation
        self.spectrum_props.slots_needed = 2
        self.helpers.engine_props['guard_slots'] = 1
        self.assertTrue(self.helpers.check_super_channels(open_slots_matrix, flag=''))

        # Test when there is no valid allocation (e.g., due to forced index)
        self.spectrum_props.forced_index = 5
        self.assertTrue(self.helpers.check_super_channels(open_slots_matrix, flag='forced_index'))

        # Test when the allocation method is last_fit
        self.helpers.engine_props['allocation_method'] = 'last_fit'
        self.assertFalse(self.helpers.check_super_channels(open_slots_matrix, flag=''))

        # Test when no super-channel can satisfy the request
        self.spectrum_props.slots_needed = 10
        self.assertFalse(self.helpers.check_super_channels(open_slots_matrix, flag=''))

    def test_check_free_spectrum_no_guard_band(self):
        """Test check_free_spectrum with guard band set to zero."""
        self.helpers.curr_band = 'c'
        self.helpers.core_num = 0
        self.helpers.start_index = 0
        self.helpers.end_index = 0
        self.spectrum_props.slots_needed = 1
        self.helpers.engine_props['guard_slots'] = 0

        result = self.helpers._check_free_spectrum((1, 2), (2, 1))
        self.assertTrue(result)

    def test_check_super_channels_no_guard_band(self):
        """Test check_super_channels method with guard band set to zero."""
        open_slots_matrix = [[0, 1, 2, 3, 4], [5, 6, 7, 8]]
        self.helpers.curr_band = 'c'
        self.helpers.core_num = 0
        self.spectrum_props.slots_needed = 2
        self.helpers.engine_props['guard_slots'] = 0

        # Test when there is a valid allocation
        self.assertTrue(self.helpers.check_super_channels(open_slots_matrix, flag=''))

        # Test when no super-channel can satisfy the request
        self.spectrum_props.slots_needed = 10
        self.assertFalse(self.helpers.check_super_channels(open_slots_matrix, flag=''))

    def test_check_other_links_no_guard_band(self):
        """Test the check_other_links method with guard band set to zero."""
        self.helpers.curr_band = 'c'
        self.helpers.core_num = 0
        self.helpers.start_index = 0
        self.helpers.end_index = 5
        self.helpers.engine_props['guard_slots'] = 0

        # Simulate a multi-link path where spectrum is free
        self.spectrum_props.is_free = False
        self.helpers.check_other_links()
        self.assertTrue(self.spectrum_props.is_free)

    def test_check_free_spectrum_l_band_only(self):
        """Test the _check_free_spectrum method when only the 'l' band is available."""
        # Fill the 'c' band and make the 'l' band available
        self.sdn_props.net_spec_dict[(1, 2)]['cores_matrix']['c'][0][:] = 1  # Fill 'c' band
        self.sdn_props.net_spec_dict[(1, 2)]['cores_matrix']['l'] = np.zeros((2, 10))  # 'l' band is free
        self.helpers.curr_band = 'l'  # Test with 'l' band
        self.helpers.core_num = 0
        self.helpers.start_index = 0
        self.helpers.end_index = 5
        self.spectrum_props.slots_needed = 6
        self.helpers.engine_props['guard_slots'] = 0

        result = self.helpers._check_free_spectrum((1, 2), (2, 1))
        self.assertTrue(result)

    def test_check_super_channels_s_band_only(self):
        """Test the check_super_channels method when only the 's' band is available."""
        open_slots_matrix = [[0, 1, 2, 3, 4], [5, 6, 7, 8]]
        # Fill 'c' and 'l' bands, leaving 's' band free
        self.sdn_props.net_spec_dict[(1, 2)]['cores_matrix']['c'][0][:] = 1
        self.sdn_props.net_spec_dict[(1, 2)]['cores_matrix']['l'][0][:] = 1

        self.helpers.curr_band = 's'
        self.helpers.core_num = 0
        self.spectrum_props.slots_needed = 2
        self.helpers.engine_props['guard_slots'] = 1

        # Test with valid allocation in the 's' band
        self.assertTrue(self.helpers.check_super_channels(open_slots_matrix, flag=''))

    def test_check_other_links_l_band_only(self):
        """Test the check_other_links method when only the 'l' band is free."""
        # Fill the 'c' band and make the 'l' band available
        self.sdn_props.net_spec_dict[(1, 2)]['cores_matrix']['c'][0][:] = 1
        self.sdn_props.net_spec_dict[(2, 3)]['cores_matrix']['c'][0][:] = 1

        self.helpers.curr_band = 'l'
        self.helpers.core_num = 0
        self.helpers.start_index = 0
        self.helpers.end_index = 5
        self.spectrum_props.path_list = [1, 2, 3]
        self.spectrum_props.slots_needed = 6
        self.helpers.engine_props['guard_slots'] = 0

        self.helpers.check_other_links()
        self.assertTrue(self.spectrum_props.is_free)


if __name__ == '__main__':
    unittest.main()
