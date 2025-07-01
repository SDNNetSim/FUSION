Data Arguments
==============

The following table summarizes the configuration details for the simulator. It shows the maximum supported request
length and the number of spectral slots needed for each combination of modulation scheme (QPSK, 16-QAM, 64-QAM) and
reach (the maximum value in kilometers a request can be allocated to). The table is separated by two sections,
"Yue's Assumptions" and "Arash's Assumptions", reflecting the two supported simulation models.

References to these assumptions can be found ``here``.


.. list-table:: Yue's Assumptions
   :widths: 25 25 25
   :header-rows: 1

   * - (Bandwidth, Modulation Format)
     - Slots Needed (w/o guard band)
     - Maximum Reach (KM)
   * - (25, QPSK)
     - 1
     - 22160
   * - (25, 16-QAM)
     - 1
     - 9500
   * - (25, 64-QAM)
     - 1
     - 3664
   * - (50, QPSK)
     - 2
     - 11080
   * - (50, 16-QAM)
     - 1
     - 4750
   * - (50, 64-QAM)
     - 1
     - 1832
   * - (100, QPSK)
     - 4
     - 5540
   * - (100, 16-QAM)
     - 2
     - 2375
   * - (100, 64-QAM)
     - 2
     - 916
   * - (200, QPSK)
     - 8
     - 2770
   * - (200, 16-QAM)
     - 4
     - 1187
   * - (200, 64-QAM)
     - 3
     - 458
   * - (400, QPSK)
     - 16
     - 1385
   * - (400, 16-QAM)
     - 8
     - 594
   * - (400, 64-QAM)
     - 6
     - 229

.. list-table:: Arash's Assumptions
   :widths: 25 25 25
   :header-rows: 1

   * - (Bandwidth, Modulation Format)
     - Slots Needed (w/o guard band)
     - Maximum Reach (KM)
   * - (25, QPSK)
     - 1
     - 20759
   * - (25, 16-QAM)
     - 1
     - 9295
   * - (25, 64-QAM)
     - 1
     - 3503
   * - (50, QPSK)
     - 2
     - 10380
   * - (50, 16-QAM)
     - 1
     - 4648
   * - (50, 64-QAM)
     - 1
     - 1752
   * - (100, QPSK)
     - 3
     - 5190
   * - (100, 16-QAM)
     - 2
     - 2324
   * - (100, 64-QAM)
     - 1
     - 876
   * - (200, QPSK)
     - 5
     - 2595
   * - (200, 16-QAM)
     - 3
     - 1162
   * - (200, 64-QAM)
     - 2
     - 438
   * - (400, QPSK)
     - 10
     - 1298
   * - (400, 16-QAM)
     - 5
     - 581
   * - (400, 64-QAM)
     - 4
     - 219
