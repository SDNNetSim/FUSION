import numpy as np

engine_props = {
    "topology_info": {
        "links": {
            "link_id": {
                "fiber": {
                    "non_linearity": 1.27e-3,
                    "attenuation": 0.001,
                    "dispersion": 100,
                },
                "span_length": 80,
                "length": 320,
            }
        }
    },
    "requested_xt": {"QPSK": -30},
    "phi": {"QPSK": 1.0},
    "bw_per_slot": 12.5,
    "input_power": 5.0,
    "spectral_slots": 4,
    "egn_model": True,
    "xt_noise": False,
}

cores_matrix = np.array(
    [
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
    ]
)
sdn_props = {
    "net_spec_dict": {
        ("source", "dest"): {
            "cores_matrix": cores_matrix,
            "link_num": "link_id",
        },
        ("source", "intermediate"): {
            "cores_matrix": cores_matrix,
            "link_num": "link_id",
        },
        ("intermediate", "dest"): {
            "cores_matrix": cores_matrix,
            "link_num": "link_id",
        },
        ("dest", "intermediate"): {
            "cores_matrix": cores_matrix,
            "link_num": "link_id",
        },
    }
}

snr_props = {
    "center_freq": 193.1e12,
    "link_dict": {
        "attenuation": 0.001,
        "dispersion": 16.2,
        "bending_radius": 5.0,
        "mode_coupling_co": 10.0,
        "propagation_const": 5.0,
        "core_pitch": 9.0,
    },
    "length": 80,
    "center_psd": 2e-10,
    "bandwidth": 50,
    "req_bit_rate": 10,
    "sci_psd": 2.4674667424119302e-18,
    "xci_psd": 4.39444915467244e-20,
    "mu_param": 7.701030231387541e-06,
    "plank": 6.62607015e-34,
    "light_frequency": 193100000000000.0,
    "nsp": 1.58,
    "req_snr": 15.0,
    "num_span": 0.01,
}
