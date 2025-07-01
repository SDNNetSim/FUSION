SNR Arguments
===============

Understanding signal-to-noise ratio (OSNR) is critical for optimizing performance in fiber optic communication systems.
OSNR reflects the strength of the desired signal compared to unwanted noise. This reference guide delves into the
various arguments that influence OSNR calculations in optical networks. By understanding these factors,
you'll gain valuable insights for designing and maintaining high-fidelity optical links.

.. automodule:: arg_scripts.snr_args
    :members:
    :undoc-members:

SNRProps Attributes
-------------------

.. list-table::
   :header-rows: 1

   * - Attribute
     - Type
     - Description

   * - light_frequency
     - float
     - Central optical carrier frequency (Hz)

   * - plank
     - float
     - Planck's constant (Joule·seconds)

   * - req_bit_rate
     - float
     - Target request bit rate (Gbps)

   * - req_snr
     - float
     - Required signal-to-noise ratio (dB)

   * - nsp
     - float
     - Noise spectral density

   * - center_freq
     - float
     - Center frequency of the current request

   * - bandwidth
     - float
     - Bandwidth for the current request

   * - center_psd
     - float
     - Centered power spectral density

   * - mu_param
     - float
     - Mu parameter used in PSD calculations

   * - sci_psd
     - float
     - Self-channel interference PSD

   * - xci_psd
     - float
     - Cross-channel interference PSD

   * - length
     - float
     - Span length

   * - num_span
     - int
     - Number of spans in path

   * - link_dict
     - dict
     - Dictionary of link lengths and related metrics

   * - mod_format_mapping_dict
     - dict
     - Mapping from numeric format index to modulation name

   * - bw_mapping_dict
     - dict
     - Bandwidth (Gbps) associated with each modulation format
