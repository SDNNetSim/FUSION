
# Flexible Unified System for Intelligent Optical Networking (FUSION)

## About This Project

Welcome to the FUSION, an open-source venture into the future of networking! Our core focus is on simulating Software Defined Elastic Optical Networks (SD-EONs), a cutting-edge approach that promises to revolutionize how data is transmitted over optical fibers. But that's just the beginning. We envision the SDN Simulator as a versatile simulation framework that can evolve to simulate a wide array of networking paradigms, now including the integration of artificial intelligence to enhance network optimization, performance, and decision-making processes.

We need your insight and creativity! The true strength of open-source lies in community collaboration. Join us in pioneering the networks of tomorrow by contributing your unique simulations and features. Your expertise in AI and networking can help shape the future of this field.

## Getting Started

### Supported Operating Systems

- Ubuntu version 20.04+
- Fedora 37+
- Windows 11

### Supported Programming Languages

- Python version 3.11+

### Installation

To get started with the FUSION, follow these steps to set up your environment:

1. Navigate to the desired directory you'd like to clone the repo to:
   ```
   cd /your/desired/path
   ```
2. Clone the repository:
   ```
   git clone git@github.com:SDNNetSim/FUSION.git
   ```
3. Change into the project directory:
   ```
   cd FUSION
   ```
4. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
5. Navigate to the docs directory and generate the Sphinx documentation:

   On Unix:
   ```
   cd docs
   make html
   ```
   On Windows:
   ```
   cd docs
   .\make html
   ```
6. Finally, navigate to `_build/html/` and open `index.html` in a browser of your choice to view the documentation.

## Standards and Guidelines

To maintain the quality and consistency of the codebase, we adhere to the following standards and guidelines:

1. **Commit Formatting**: Follow the commit format specified [here](https://gist.github.com/robertpainsi/b632364184e70900af4ab688decf6f53).
2. **Code Style**: All code should follow the [PEP 8](https://peps.python.org/pep-0008/) coding style guidelines.
3. **Versioning**: Use the [semantic versioning system](https://semver.org/) for all git tags.
4. **Coding Guidelines**: Adhere to the team's [coding guidelines document](https://github.com/SDNNetSim/sdn_simulator/blob/main/CONTRIBUTING.md).
5. **Unit Testing**: Each unit test should follow the [community unit testing guidelines](https://pylonsproject.org/community-unit-testing-guidelines.html).

## Contributors

This project is brought to you by the efforts of Arash Rezaee, Ryan McCann, Suvhasis Mukhopadhyay, Kojo Bempah, and 
Kimberly Tice. We welcome contributions from the community to help make this project even better!
