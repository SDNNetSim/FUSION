# I/O Module TODOs

## High Priority

*No high priority items currently.*

## Medium Priority

### Standardize Network Topology File Names (v6.0/v6.1)
- **Issue**: Network name to file path mapping uses inconsistent naming conventions
- **Files**: `structure.py:86`
- **Current State**: Manual mapping dictionary where input names don't match file names:
  ```python
  network_files = {
      "USNet": "us_network.txt",
      "NSFNet": "nsf_network.txt",
      "Pan-European": "europe_network.txt",
      "USbackbone60": "USB6014.txt",
      "Spainbackbone30": "SPNB3014.txt",
      ...
  }
  ```
- **Solution**: Rename topology files to match input names directly (e.g., `USNet.txt`, `NSFNet.txt`) to eliminate the mapping dictionary and allow automatic file discovery

## Low Priority

*No low priority items currently.*
