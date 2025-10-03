#!/usr/bin/env python3
"""
Automated API documentation generator for FUSION.

This script automatically generates RST files for all modules in the FUSION package,
creating comprehensive API documentation with proper cross-references and structure.

Usage:
    python autogen.py [--clean] [--verbose]

Options:
    --clean     Remove all generated files before regenerating
    --verbose   Print detailed progress information
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path to import FUSION
sys.path.insert(0, os.path.abspath(".."))

# Import after path modification
try:
    import fusion
except ImportError as e:
    print(f"Warning: Could not import fusion package: {e}")
    print("Some features may be limited.")
    fusion = None


class APIDocGenerator:
    """Generate RST files for FUSION API documentation."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.api_dir = Path(__file__).parent / "api"
        self.api_dir.mkdir(exist_ok=True)

        # Define package structure
        self.packages = {
            "core": {
                "title": "Core",
                "description": (
                    "Core simulation engine, SDN controller, and network management"
                ),
                "modules": [
                    "fusion.core.engine",
                    "fusion.core.sdn_controller",
                    "fusion.core.request_generator",
                    "fusion.core.network",
                ],
            },
            "modules": {
                "title": "Modules",
                "description": "Pluggable algorithm implementations",
                "modules": [
                    "fusion.modules.routing",
                    "fusion.modules.spectrum",
                    "fusion.modules.snr",
                    "fusion.modules.ml",
                    "fusion.modules.rl",
                ],
            },
            "sim": {
                "title": "Simulation",
                "description": (
                    "High-level simulation pipelines and workflow orchestration"
                ),
                "modules": [
                    "fusion.sim.run_sim",
                    "fusion.sim.run_ml_sim",
                    "fusion.sim.run_rl_sim",
                ],
            },
            "configs": {
                "title": "Configuration",
                "description": (
                    "Configuration management, validation, and schema definition"
                ),
                "modules": [
                    "fusion.configs.config",
                    "fusion.configs.args",
                ],
            },
            "cli": {
                "title": "CLI",
                "description": "Command-line interface for running simulations",
                "modules": [
                    "fusion.cli.main",
                ],
            },
            "io": {
                "title": "I/O",
                "description": "Data generation, export, and structured file handling",
                "modules": [
                    "fusion.io.generate_data",
                    "fusion.io.structure_data",
                ],
            },
            "utils": {
                "title": "Utilities",
                "description": "General-purpose utilities",
                "modules": [
                    "fusion.utils.helpers",
                    "fusion.utils.network",
                    "fusion.utils.random",
                    "fusion.utils.spectrum",
                ],
            },
            "interfaces": {
                "title": "Interfaces",
                "description": "Abstract base classes for extensibility",
                "modules": [
                    "fusion.interfaces.router",
                    "fusion.interfaces.spectrum",
                    "fusion.interfaces.snr",
                ],
            },
            "analysis": {
                "title": "Analysis",
                "description": "Data analysis and statistical tools",
                "modules": [
                    "fusion.analysis.stats",
                ],
            },
            "reporting": {
                "title": "Reporting",
                "description": "Results reporting and export",
                "modules": [
                    "fusion.reporting.export",
                ],
            },
            "visualization": {
                "title": "Visualization",
                "description": "Plotting and visualization tools",
                "modules": [
                    "fusion.visualization.plot",
                ],
            },
            "unity": {
                "title": "Unity",
                "description": "Integration utilities",
                "modules": [
                    "fusion.unity.helpers",
                ],
            },
        }

    def log(self, message: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(f"[APIDocGen] {message}")

    def generate_module_rst(self, module_path: str, section_title: str) -> str:
        """Generate RST content for a single module."""
        module_name = module_path.split(".")[-1]

        rst_content = f"""
{module_name}
{"=" * len(module_name)}

.. automodule:: {module_path}
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

"""
        return rst_content

    def generate_package_rst(self, package_key: str, package_info: dict) -> str:
        """Generate RST content for a package page."""
        title = package_info["title"]
        description = package_info["description"]

        rst_content = f"""
{title}
{"=" * len(title)}

{description}

.. contents:: Table of Contents
   :local:
   :depth: 2

"""

        # Add each module section
        for module_path in package_info["modules"]:
            module_name = module_path.split(".")[-1].replace("_", " ").title()

            rst_content += f"""
{module_name}
{"-" * len(module_name)}

.. automodule:: {module_path}
   :members:
   :undoc-members:
   :show-inheritance:

"""

        return rst_content

    def generate_index_rst(self) -> str:
        """Generate the main API index page."""
        rst_content = """
=============
API Reference
=============

Complete API documentation for all FUSION modules.

.. contents:: Table of Contents
   :local:
   :depth: 2

Package Overview
================

FUSION is organized into the following main packages:

Core Packages
-------------

:doc:`core`
   Core simulation engine, SDN controller, routing, spectrum assignment,
   and SNR measurements

:doc:`modules`
   Pluggable algorithm implementations:

   * **Routing**: Path computation algorithms
   * **Spectrum**: Spectrum assignment strategies
   * **SNR**: Signal-to-noise ratio calculation
   * **ML**: Machine learning algorithms and utilities
   * **RL**: Reinforcement learning agents, algorithms, and environments

:doc:`sim`
   High-level simulation pipelines and workflow orchestration

Configuration & CLI
-------------------

:doc:`configs`
   Configuration management, validation, and schema definition

:doc:`cli`
   Command-line interface for running simulations and training

Data & I/O
----------

:doc:`io`
   Data generation, export, and structured file handling

:doc:`utils`
   General-purpose utilities (logging, network, random, spectrum operations)

Advanced Features
-----------------

:doc:`interfaces`
   Abstract base classes for creating custom plugins

:doc:`analysis`
   Statistical analysis and data processing tools

:doc:`reporting`
   Results reporting, export formats, and data serialization

:doc:`visualization`
   Plotting, charting, and visual analysis tools

:doc:`unity`
   Integration and helper utilities

Complete Module List
=====================

.. toctree::
   :maxdepth: 2

"""

        # Add all package references
        for package_key in self.packages.keys():
            rst_content += f"   {package_key}\n"

        rst_content += """

Indices
=======

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
"""

        return rst_content

    def clean_generated_files(self):
        """Remove all previously generated RST files."""
        self.log("Cleaning previously generated files...")

        for file in self.api_dir.glob("*.rst"):
            if file.name != "index.rst":
                file.unlink()
                self.log(f"  Removed: {file.name}")

    def generate_all(self):
        """Generate all API documentation files."""
        self.log("Starting API documentation generation...")

        # Generate index
        self.log("Generating index.rst...")
        index_path = self.api_dir / "index.rst"
        index_path.write_text(self.generate_index_rst())

        # Generate package pages
        for package_key, package_info in self.packages.items():
            self.log(f"Generating {package_key}.rst...")
            package_path = self.api_dir / f"{package_key}.rst"
            package_path.write_text(
                self.generate_package_rst(package_key, package_info)
            )

        self.log(f"Successfully generated {len(self.packages) + 1} files!")

    def verify_modules(self) -> dict[str, list[str]]:
        """Verify which modules can be imported."""
        if fusion is None:
            return {"missing": ["All modules (fusion package not importable)"]}

        results = {"found": [], "missing": []}

        for package_info in self.packages.values():
            for module_path in package_info["modules"]:
                try:
                    __import__(module_path)
                    results["found"].append(module_path)
                except ImportError:
                    results["missing"].append(module_path)

        return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate FUSION API documentation")
    parser.add_argument(
        "--clean", action="store_true", help="Clean generated files before regenerating"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print detailed progress"
    )
    parser.add_argument(
        "--verify", action="store_true", help="Verify module imports only"
    )

    args = parser.parse_args()

    generator = APIDocGenerator(verbose=args.verbose)

    if args.verify:
        print("Verifying module imports...")
        results = generator.verify_modules()
        print(f"\nFound: {len(results['found'])} modules")
        print(f"Missing: {len(results['missing'])} modules")

        if results["missing"]:
            print("\nMissing modules:")
            for module in results["missing"]:
                print(f"  - {module}")

        return

    if args.clean:
        generator.clean_generated_files()

    generator.generate_all()

    print("\nAPI documentation generation complete!")
    print(f"Generated files in: {generator.api_dir}")


if __name__ == "__main__":
    main()
