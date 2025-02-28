# pylint: disable=too-few-public-methods

class SNRProps:
    """
    Main properties used for the snr_measurements.py script.
    """

    def __init__(self):
        self.light_frequency = 1.9341 * 10 ** 14  # Center light frequency
        self.plank = 6.62607004e-34  # Plank's constant
        self.req_bit_rate = 12.5  # Request bit rate
        self.req_snr = 8.5  # Request signal to noise ratio value
        self.nsp = 1.8  # Noise spectral density

        self.center_freq = None  # Center frequency for current request
        self.bandwidth = None  # Bandwidth for current request
        self.center_psd = None  # Center power spectral density for current request
        self.mu_param = None  # Mu parameter for calculating PSD
        self.sci_psd = None  # Self-channel interference PSD
        self.xci_psd = None  # Cross-channel interference PSD
        self.length = None  # Length of a current span
        self.num_span = None  # Number of span

        self.link_dict = None  # Dictionary of links for calculating various metrics
        self.mod_format_mapping_dict = {6: "64-QAM", 5: "32-QAM", 4: "16-QAM", 3: "8-QAM", 2: "QPSK", 1: "BPSK"} # Dictionary of Modulation formats for precalculated SNR
        self.bw_mapping_dict = {"64-QAM": 600, "32-QAM": 500, "16-QAM": 400, "8-QAM": 300, "QPSK": 200, "BPSK": 100} # Dictionary of Modulation formats to calculate the supported bit rate for fixed grid
        self.file_mapping_dict = {
            'USbackbone60': {
                                'multi_fiber': {'mf': 'MF-USB6014-MF.npy', 'gsnr': 'GSNR-USB6014-MF.npy'},
                                (2, 4): {'mf': 'MF-USB6014-MCF4-C2.npy', 'gsnr': 'GSNR-USB6014-MCF4-C2.npy'},
                                (6, 7): {'mf': 'MF-USB6014-MCF7-C6.npy', 'gsnr': 'GSNR-USB6014-MCF7-C6.npy'},
                                (3, 7): {'mf': 'MF-USB6014-MCF7-C3.npy', 'gsnr': 'GSNR-USB6014-MCF7-C3.npy'},
                                (2, 13): {'mf': 'MF-USB6014-MCF13-C2.npy', 'gsnr': 'GSNR-USB6014-MCF13-C2.npy'},
                                (5, 13): {'mf': 'MF-USB6014-MCF13-C5.npy', 'gsnr': 'GSNR-USB6014-MCF13-C5.npy'},
                                (6, 13): {'mf': 'MF-USB6014-MCF13-C6.npy', 'gsnr': 'GSNR-USB6014-MCF13-C6.npy'},
                                (6, 19): {'mf': 'MF-USB6014-MCF19-C6.npy', 'gsnr': 'GSNR-USB6014-MCF19-C6.npy'},
                                (3, 19): {'mf': 'MF-USB6014-MCF19-C3.npy', 'gsnr': 'GSNR-USB6014-MCF19-C3.npy'},
                                (4, 19): {'mf': 'MF-USB6014-MCF19-C4.npy', 'gsnr': 'GSNR-USB6014-MCF19-C4.npy'},
                            },
            'Spainbackbone30':{

                            'multi_fiber': {'mf': 'MF-SPNB3014-MF.npy', 'gsnr': 'GSNR-SPNB3014-MF.npy'},
                            (2, 4): {'mf': 'MF-SPNB3014-MCF4-C2.npy', 'gsnr': 'GSNR-SPNB3014-MCF4-C2.npy'},
                            (6, 7): {'mf': 'MF-SPNB3014-MCF7-C6.npy', 'gsnr': 'GSNR-SPNB3014-MCF7-C6.npy'},
                            (3, 7): {'mf': 'MF-SPNB3014-MCF7-C3.npy', 'gsnr': 'GSNR-SPNB3014-MCF7-C3.npy'},
                            (2, 13): {'mf': 'MF-SPNB3014-MCF13-C2.npy', 'gsnr': 'GSNR-SPNB3014-MCF13-C2.npy'},
                            (5, 13): {'mf': 'MF-SPNB3014-MCF13-C5.npy', 'gsnr': 'GSNR-SPNB3014-MCF13-C5.npy'},
                            (6, 13): {'mf': 'MF-SPNB3014-MCF13-C6.npy', 'gsnr': 'GSNR-SPNB3014-MCF13-C6.npy'},
                            (6, 19): {'mf': 'MF-SPNB3014-MCF19-C6.npy', 'gsnr': 'GSNR-SPNB3014-MCF19-C6.npy'},
                            (3, 19): {'mf': 'MF-SPNB3014-MCF19-C3.npy', 'gsnr': 'GSNR-SPNB3014-MCF19-C3.npy'},
                            (4, 19): {'mf': 'MF-SPNB3014-MCF19-C4.npy', 'gsnr': 'GSNR-SPNB3014-MCF19-C4.npy'},
                        }
                            }

    def __repr__(self):
        return f"SNRProps({self.__dict__})"
