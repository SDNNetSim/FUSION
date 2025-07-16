# Flexible Unified System for Intelligent Optical Networking (FUSION)

## About This Project

Welcome to **FUSION**, an open-source venture into the future of networking! Our core focus is on simulating **Software Defined Elastic Optical Networks (SD-EONs)**, a cutting-edge approach that promises to revolutionize how data is transmitted over optical fibers. But that's just the beginning. We envision FUSION as a versatile simulation framework that can evolve to simulate a wide array of networking paradigms, now including the integration of **artificial intelligence** to enhance network optimization, performance, and decision-making processes.

We need your insight and creativity! The true strength of open-source lies in community collaboration. Join us in pioneering the networks of tomorrow by contributing your unique simulations and features. Your expertise in AI and networking can help shape the future of this field.

## Getting Started

### Supported Operating Systems

- macOS (requires manual compilation steps)
- Ubuntu 20.04+
- Fedora 37+
- Windows 11

### Supported Programming Languages

- Python 3.11.X

---

## Installation Instructions

To get started with FUSION, first clone the repository and create a Python 3.11 virtual environment:

```bash
# Navigate to your desired directory
cd /your/desired/path

# Clone the repository
git clone git@github.com:SDNNetSim/FUSION.git
cd FUSION

# Create and activate a Python 3.11 virtual environment
python3.11 -m venv venv
source venv/bin/activate
```

Next, follow the specific instructions for your operating system.

---

### macOS Installation

Installation on macOS is a multi-step process that requires compiling packages from source. Please follow these steps carefully.

**Step 1: Install Prerequisites**

Ensure you have Apple’s Command Line Tools installed:

```bash
xcode-select --install
```

**Step 2: Install PyTorch**

```bash
pip install torch==2.2.2
```

**Step 3: Install PyTorch Geometric (PyG) Packages**

These packages require special flags to compile correctly on macOS:

```bash
MACOSX_DEPLOYMENT_TARGET=10.15 pip install --no-build-isolation torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.2.2+cpu.html
```

**Step 4: Install Remaining Dependencies**

```bash
pip install -r requirements.txt
```

---

### Linux & Windows Installation

Installation on Linux and Windows is more straightforward.

**Step 1: Install PyTorch**

```bash
pip install torch==2.2.2
```

**Step 2: Install All Other Dependencies**

```bash
pip install -r requirements.txt
```

---

## Generating the Documentation

After installing the dependencies, you can generate the Sphinx documentation.

Navigate to the docs directory:

```bash
cd docs
```

Build the HTML documentation:

On macOS/Linux:

```bash
make html
```

On Windows:

```powershell
.\make.bat html
```

Finally, navigate to `_build/html/` and open `index.html` in a browser of your choice to view the documentation.

---

## Standards and Guidelines

To maintain the quality and consistency of the codebase, we adhere to the following standards and guidelines:

1. **Commit Formatting**: Follow the commit format specified [here](https://gist.github.com/robertpainsi/b632364184e70900af4ab688decf6f53).
2. **Code Style**: All code should follow the [PEP 8](https://peps.python.org/pep-0008/) coding style guidelines.
3. **Versioning**: Use the [semantic versioning system](https://semver.org/) for all git tags.
4. **Coding Guidelines**: Adhere to the team's [coding guidelines document](https://github.com/SDNNetSim/sdn_simulator/blob/main/CONTRIBUTING.md).
5. **Unit Testing**: Each unit test should follow the [community unit testing guidelines](https://pylonsproject.org/community-unit-testing-guidelines.html).

---

## Contributors

This project is brought to you by the efforts of **Arash Rezaee**, **Ryan McCann**, and **Vinod M. Vokkarane**. We welcome contributions from the community to help make this project even better!

---

## 📖 How to Cite This Work

If you use FUSION in your research, please cite the following paper:

R. McCann, A. Rezaee, and V. M. Vokkarane,  
"FUSION: A Flexible Unified Simulator for Intelligent Optical Networking,"  
*2024 IEEE International Conference on Advanced Networks and Telecommunications Systems (ANTS)*, Guwahati, India, 2024, pp. 1-6.  
DOI: [10.1109/ANTS63515.2024.10898199](https://doi.org/10.1109/ANTS63515.2024.10898199)

### 📄 BibTeX

```bibtex
@INPROCEEDINGS{10898199,
  author={McCann, Ryan and Rezaee, Arash and Vokkarane, Vinod M.},
  booktitle={2024 IEEE International Conference on Advanced Networks and Telecommunications Systems (ANTS)}, 
  title={FUSION: A Flexible Unified Simulator for Intelligent Optical Networking}, 
  year={2024},
  pages={1-6},
  doi={10.1109/ANTS63515.2024.10898199}
}
```
