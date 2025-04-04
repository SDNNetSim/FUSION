YUE_MOD_ASSUMPTIONS = {
    '25': {'QPSK': {'max_length': 22160, 'slots_needed': 1}, '16-QAM': {'max_length': 9500, 'slots_needed': 1},
           '64-QAM': {'max_length': 3664, 'slots_needed': 1}},
    '50': {'QPSK': {'max_length': 11080, 'slots_needed': 2}, '16-QAM': {'max_length': 4750, 'slots_needed': 1},
           '64-QAM': {'max_length': 1832, 'slots_needed': 1}},
    '100': {'QPSK': {'max_length': 5540, 'slots_needed': 4}, '16-QAM': {'max_length': 2375, 'slots_needed': 2},
            '64-QAM': {'max_length': 916, 'slots_needed': 2}},
    '200': {'QPSK': {'max_length': 2770, 'slots_needed': 8}, '16-QAM': {'max_length': 1187, 'slots_needed': 4},
            '64-QAM': {'max_length': 458, 'slots_needed': 3}},
    '400': {'QPSK': {'max_length': 1385, 'slots_needed': 16}, '16-QAM': {'max_length': 594, 'slots_needed': 8},
            '64-QAM': {'max_length': 229, 'slots_needed': 6}},
}

# ARASH_MOD_ASSUMPTIONS = {
#     "25": {"QPSK": {"max_length": 20759, "slots_needed": 1}, "16-QAM": {"max_length": 9295, "slots_needed": 1},
#            "64-QAM": {"max_length": 3503, "slots_needed": 1}},

#     "50": {"QPSK": {"max_length": 10380, "slots_needed": 2}, "16-QAM": {"max_length": 4648, "slots_needed": 1},
#            "64-QAM": {"max_length": 1752, "slots_needed": 1}},

#     "100": {"QPSK": {"max_length": 5190, "slots_needed": 3}, "16-QAM": {"max_length": 2324, "slots_needed": 2},
#             "64-QAM": {"max_length": 876, "slots_needed": 1}},

#     "200": {"QPSK": {"max_length": 2595, "slots_needed": 5}, "16-QAM": {"max_length": 1162, "slots_needed": 3},
#             "64-QAM": {"max_length": 438, "slots_needed": 2}},

#     "400": {"QPSK": {"max_length": 1298, "slots_needed": 10}, "16-QAM": {"max_length": 581, "slots_needed": 5},
#             "64-QAM": {"max_length": 219, "slots_needed": 4}}
# }
"""
ARASH_MOD_ASSUMPTIONS = {
    "100": {
        "BPSK": {
            "max_length": 41460,
            "slots_needed": 8
        },
        "QPSK": {
            "max_length": 41210,
            "slots_needed": 4
        },
        "8-QAM": {
            "max_length": 41000,
            "slots_needed": 3
        },
        "16-QAM": {
            "max_length": 40800,
            "slots_needed": 3
        },
        "32-QAM": {
            "max_length": 40670,
            "slots_needed": 3
        },
        "64-QAM": {
            "max_length": 40660,
            "slots_needed": 3
        }
    },
    "200": {
        "BPSK": {
            "max_length": 41460,
            "slots_needed": 15
        },
        "QPSK": {
            "max_length": 41210,
            "slots_needed": 8
        },
        "8-QAM": {
            "max_length": 41000,
            "slots_needed": 5
        },
        "16-QAM": {
            "max_length": 40800,
            "slots_needed": 4
        },
        "32-QAM": {
            "max_length": 40670,
            "slots_needed": 3
        },
        "64-QAM": {
            "max_length": 40660,
            "slots_needed": 3
        }
    },
    "300": {
        "BPSK": {
            "max_length": 41460,
            "slots_needed": 22
        },
        "QPSK": {
            "max_length": 41210,
            "slots_needed": 11
        },
        "8-QAM": {
            "max_length": 41000,
            "slots_needed": 8
        },
        "16-QAM": {
            "max_length": 40800,
            "slots_needed": 6
        },
        "32-QAM": {
            "max_length": 40670,
            "slots_needed": 5
        },
        "64-QAM": {
            "max_length": 40660,
            "slots_needed": 4
        }
    },
    "400": {
        "BPSK": {
            "max_length": 41460,
            "slots_needed": 29
        },
        "QPSK": {
            "max_length": 41210,
            "slots_needed": 15
        },
        "8-QAM": {
            "max_length": 41000,
            "slots_needed": 10
        },
        "16-QAM": {
            "max_length": 40800,
            "slots_needed": 8
        },
        "32-QAM": {
            "max_length": 40670,
            "slots_needed": 6
        },
        "64-QAM": {
            "max_length": 40660,
            "slots_needed": 5
        }
    },
    "500": {
        "BPSK": {
            "max_length": 41460,
            "slots_needed": 36
        },
        "QPSK": {
            "max_length": 41210,
            "slots_needed": 18
        },
        "8-QAM": {
            "max_length": 41000,
            "slots_needed": 12
        },
        "16-QAM": {
            "max_length": 40800,
            "slots_needed": 9
        },
        "32-QAM": {
            "max_length": 40670,
            "slots_needed": 8
        },
        "64-QAM": {
            "max_length": 40660,
            "slots_needed": 6
        }
    },
    "600": {
        "BPSK": {
            "max_length": 41460,
            "slots_needed": 43
        },
        "QPSK": {
            "max_length": 41210,
            "slots_needed": 22
        },
        "8-QAM": {
            "max_length": 41000,
            "slots_needed": 15
        },
        "16-QAM": {
            "max_length": 40800,
            "slots_needed": 11
        },
        "32-QAM": {
            "max_length": 40670,
            "slots_needed": 9
        },
        "64-QAM": {
            "max_length": 40660,
            "slots_needed": 8
        }
    },
    "700": {
        "BPSK": {
            "max_length": 41460,
            "slots_needed": 50
        },
        "QPSK": {
            "max_length": 41210,
            "slots_needed": 25
        },
        "8-QAM": {
            "max_length": 41000,
            "slots_needed": 17
        },
        "16-QAM": {
            "max_length": 40800,
            "slots_needed": 13
        },
        "32-QAM": {
            "max_length": 40670,
            "slots_needed": 10
        },
        "64-QAM": {
            "max_length": 40660,
            "slots_needed": 9
        }
    },
    "800": {
        "BPSK": {
            "max_length": 41460,
            "slots_needed": 58
        },
        "QPSK": {
            "max_length": 41210,
            "slots_needed": 29
        },
        "8-QAM": {
            "max_length": 41000,
            "slots_needed": 20
        },
        "16-QAM": {
            "max_length": 40800,
            "slots_needed": 15
        },
        "32-QAM": {
            "max_length": 40670,
            "slots_needed": 12
        },
        "64-QAM": {
            "max_length": 40660,
            "slots_needed": 10
        }
    }
}

"""


ARASH_MOD_ASSUMPTIONS = {
    "100": {
        "BPSK": {"max_length": 41460, "slots_needed": 1},
        "QPSK": {"max_length": 41210, "slots_needed": 1},
        "8-QAM": {"max_length": 41000, "slots_needed": 1},
        "16-QAM": {"max_length": 40800, "slots_needed": 1},
        "32-QAM": {"max_length": 40670 , "slots_needed": 1},
        "64-QAM": {"max_length": 40660, "slots_needed": 1}
    },
    "200": {
        "BPSK": {"max_length": 41460, "slots_needed": 1},
        "QPSK": {"max_length": 41210, "slots_needed": 1},
        "8-QAM": {"max_length": 41000, "slots_needed": 1},
        "16-QAM": {"max_length": 40800, "slots_needed": 1},
        "32-QAM": {"max_length": 40670 , "slots_needed": 1},
        "64-QAM": {"max_length": 40660, "slots_needed": 1}
    },
    "300": {
        "BPSK": {"max_length": 41460, "slots_needed": 1},
        "QPSK": {"max_length": 41210, "slots_needed": 1},
        "8-QAM": {"max_length": 41000, "slots_needed": 1},
        "16-QAM": {"max_length": 40800, "slots_needed": 1},
        "32-QAM": {"max_length": 40670 , "slots_needed": 1},
        "64-QAM": {"max_length": 40660, "slots_needed": 1}
    },
    "400": {
        "BPSK": {"max_length": 41460, "slots_needed": 1},
        "QPSK": {"max_length": 41210, "slots_needed": 1},
        "8-QAM": {"max_length": 41000, "slots_needed": 1},
        "16-QAM": {"max_length": 40800, "slots_needed": 1},
        "32-QAM": {"max_length": 40670 , "slots_needed": 1},
        "64-QAM": {"max_length": 40660, "slots_needed": 1}
    },
    "500": {
        "BPSK": {"max_length": 41460, "slots_needed": 1},
        "QPSK": {"max_length": 41210, "slots_needed": 1},
        "8-QAM": {"max_length": 41000, "slots_needed": 1},
        "16-QAM": {"max_length": 40800, "slots_needed": 1},
        "32-QAM": {"max_length": 40670 , "slots_needed": 1},
        "64-QAM": {"max_length": 40660, "slots_needed": 1}
    },
    "600": {
        "BPSK": {"max_length": 41460, "slots_needed": 1},
        "QPSK": {"max_length": 41210, "slots_needed": 1},
        "8-QAM": {"max_length": 41000, "slots_needed": 1},
        "16-QAM": {"max_length": 40800, "slots_needed": 1},
        "32-QAM": {"max_length": 40670 , "slots_needed": 1},
        "64-QAM": {"max_length": 40660, "slots_needed": 1}
    },
    "700": {
        "BPSK": {"max_length": 41460, "slots_needed": 1},
        "QPSK": {"max_length": 41210, "slots_needed": 1},
        "8-QAM": {"max_length": 41000, "slots_needed": 1},
        "16-QAM": {"max_length": 40800, "slots_needed": 1},
        "32-QAM": {"max_length": 40670 , "slots_needed": 1},
        "64-QAM": {"max_length": 40660, "slots_needed": 1}
    },
    "800": {
        "BPSK": {"max_length": 41460, "slots_needed": 1},
        "QPSK": {"max_length": 41210, "slots_needed": 1},
        "8-QAM": {"max_length": 41000, "slots_needed": 1},
        "16-QAM": {"max_length": 40800, "slots_needed": 1},
        "32-QAM": {"max_length": 40670, "slots_needed": 1},
        "64-QAM": {"max_length": 40660, "slots_needed": 1}
    }
}

# ARASH_MOD_ASSUMPTIONS = {'100': {'BPSK': {"max_length": 41460, 'slots_needed': 4},
#                                 'QPSK': {"max_length": 41210, 'slots_needed': 3},
#                                 '8-QAM': {"max_length": 41000, 'slots_needed': 3},
#                                 '16-QAM': {"max_length": 40800, 'slots_needed': 3},
#                                 '32-QAM': {"max_length": 40670, 'slots_needed': 3},
#                                 '64-QAM': {"max_length": 40660, 'slots_needed': 3}},
#                         '200': {         
#                                 'BPSK': {"max_length": 41460, 'slots_needed': 8},
#                                 'QPSK': {"max_length": 41210, 'slots_needed': 4},
#                                 '8-QAM': {"max_length": 41000, 'slots_needed': 3},
#                                 '16-QAM': {"max_length": 40800, 'slots_needed': 3},
#                                 '32-QAM': {"max_length": 40670, 'slots_needed': 3},
#                                 '64-QAM': {"max_length": 40660, 'slots_needed': 3}},
#                         '300': {
#                                 'BPSK': {"max_length": 41460, 'slots_needed': 12},
#                                 'QPSK': {"max_length": 41210, 'slots_needed': 6},
#                                 '8-QAM': {"max_length": 41000, 'slots_needed': 4},
#                                 '16-QAM': {"max_length": 40800, 'slots_needed': 3},
#                                 '32-QAM': {"max_length": 40670, 'slots_needed': 3},
#                                 '64-QAM': {"max_length": 40660, 'slots_needed': 3}},
#                         '400': {
#                                 'BPSK': {"max_length": 41460, 'slots_needed': 16},
#                                 'QPSK': {"max_length": 41210, 'slots_needed': 8},
#                                 '8-QAM': {"max_length": 41000, 'slots_needed': 6},
#                                 '16-QAM': {"max_length": 40800, 'slots_needed': 4},
#                                 '32-QAM': {"max_length": 40670, 'slots_needed': 4},
#                                 '64-QAM': {"max_length": 40660, 'slots_needed': 3}},
#                         '500': {         
#                                 'BPSK': {"max_length": 41460, 'slots_needed': 20},
#                                 'QPSK': {"max_length": 41210, 'slots_needed': 10},
#                                 '8-QAM': {"max_length": 41000, 'slots_needed': 7},
#                                 '16-QAM': {"max_length": 40800, 'slots_needed': 5},
#                                 '32-QAM': {"max_length": 40670, 'slots_needed': 4},
#                                 '64-QAM': {"max_length": 40660, 'slots_needed': 4}},
#                         '600': {
#                                 'BPSK': {"max_length": 41460, 'slots_needed': 24},
#                                 'QPSK': {"max_length": 41210, 'slots_needed': 12},
#                                 '8-QAM': {"max_length": 41000, 'slots_needed': 8},
#                                 '16-QAM': {"max_length": 40800, 'slots_needed': 6},
#                                 '32-QAM': {"max_length": 40670, 'slots_needed': 5},
#                                 '64-QAM': {"max_length": 40660, 'slots_needed': 4}},
#                         '700': {
#                                 'BPSK': {"max_length": 41460, 'slots_needed': 28},
#                                 'QPSK': {"max_length": 41210, 'slots_needed': 14},
#                                 '8-QAM': {"max_length": 41000, 'slots_needed': 10},
#                                 '16-QAM': {"max_length": 40800, 'slots_needed': 7},
#                                 '32-QAM': {"max_length": 40670, 'slots_needed': 6},
#                                 '64-QAM': {"max_length": 40660, 'slots_needed': 5}},
#                         '800': {
#                                 'BPSK': {"max_length": 41460, 'slots_needed': 32},
#                                 'QPSK': {"max_length": 41210, 'slots_needed': 16},
#                                 '8-QAM': {"max_length": 41000, 'slots_needed': 11},
#                                 '16-QAM': {"max_length": 40800, 'slots_needed': 8},
#                                 '32-QAM': {"max_length": 40670, 'slots_needed': 7},
#                                 '64-QAM': {"max_length": 40660, 'slots_needed': 6}}}
