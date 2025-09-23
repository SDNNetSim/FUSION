#!/usr/bin/env python3
"""
Setup script for FUSION - Flexible Unified System for Intelligent Optical Networking.

This package provides simulation capabilities for Software Defined Elastic Optical Networks (SD-EONs)
with artificial intelligence integration for network optimization.
"""

from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "FUSION - Flexible Unified System for Intelligent Optical Networking"

# Read requirements from requirements.txt
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    requirements = []
    
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines, comments, and pip-specific options
                if line and not line.startswith('#') and not line.startswith('--'):
                    # Handle version constraints
                    if '~=' in line:
                        # Convert ~= to >= for setuptools compatibility
                        line = line.replace('~=', '>=')
                    requirements.append(line)
    
    return requirements

setup(
    name="fusion-optical-networks",
    version="2.0.0",
    author="FUSION Development Team",
    author_email="fusion-dev@example.com",
    description="Flexible Unified System for Intelligent Optical Networking",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/fusion-dev/FUSION",
    packages=find_packages(exclude=['tests*', 'docs*', 'tools*', 'venv*']),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Networking",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
    install_requires=[
        # Core dependencies that are always needed
        "networkx>=3.2.1",
        "numpy>=1.26.3",
        "pandas>=2.2.0",
        "scikit-learn>=1.4.0",
        "scipy>=1.13.0",
        "matplotlib>=3.8.2",
        "PyQt5>=5.15.10",
        "seaborn>=0.13.2",
        "torch>=2.2.2",
        "PyYAML>=6.0.1",
        "requests>=2.32.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.3.4",
            "ruff>=0.1.0",
            "mypy>=1.0.0",
            "vulture>=2.7.0",
            "bandit>=1.7.0",
            "sphinx>=7.2.6",
            "sphinx_rtd_theme>=2.0.0",
        ],
        "rl": [
            "stable-baselines3>=2.2.1",
            "rl_zoo3>=2.2.1",
            "gymnasium>=0.29.1",
            "optuna>=3.6.1",
        ],
        "pyg": [
            # Note: PyTorch Geometric packages require special installation
            # Use install.sh script for automatic installation
            "torch-geometric>=2.6.1",
            # torch-scatter, torch-sparse, torch-cluster, torch-spline-conv
            # are installed separately with platform-specific flags
        ],
        "all": [
            "stable-baselines3>=2.2.1",
            "rl_zoo3>=2.2.1", 
            "gymnasium>=0.29.1",
            "optuna>=3.6.1",
            "torch-geometric>=2.6.1",
        ],
    },
    entry_points={
        "console_scripts": [
            "fusion-sim=fusion.cli.run_sim:main",
            "fusion-train=fusion.cli.run_train:main",
            "fusion-gui=fusion.cli.run_gui:main",
        ],
    },
    package_data={
        "fusion": [
            "configs/templates/*.ini",
            "configs/schemas/*.json",
            "gui/media/*.png",
            "unity/bash_scripts/*.sh",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="optical networks, machine learning, simulation, reinforcement learning, software defined networks",
    project_urls={
        "Bug Reports": "https://github.com/fusion-dev/FUSION/issues",
        "Source": "https://github.com/fusion-dev/FUSION",
        "Documentation": "https://fusion-optical-networks.readthedocs.io/",
    },
)