# EMLDDMM

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)
[![Documentation Status](https://readthedocs.org/projects/emlddmm/badge/?version=latest)](https://twardlab.github.io/emlddmm/build/html/index.html)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Expectation Maximization Large Deformation Diffeomorphic Metric Mapping**

EMLDDMM is a robust image registration framework designed to align datasets with differing contrast profiles, missing tissue, or artifacts. It leverages the Expectation Maximization (EM) algorithm to handle missing data and the Large Deformation Diffeomorphic Metric Mapping (LDDMM) paradigm to ensure diffeomorphic mappings.

[**Explore the Docs »**](https://twardlab.github.io/emlddmm/build/html/index.html)
[**Web Interface »**](https://twardlab.com/reg)

---

## 🌟 Key Features

- **Robust Alignment**: Handles differing contrasts and missing data effectively.
- **Diffeomorphic Mappings**: Ensures smooth, invertible transformations.
- **Multi-Modality Support**: Efficient pipelines for registering datasets with multiple image modalities.
- **Versatile Inputs**: Supports 3D-to-3D registration and 3D-to-2D serial section alignment.
- **Standard Formats**: Works with VTK, NIfTI, NRRD, and other common medical imaging formats.

## 🚀 Quick Start

### Prerequisites

- Python 3.6+
- PyTorch (GPU acceleration recommended but not required)

### Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/xl1393/EMLDDMM.git
    cd EMLDDMM
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

### Basic Usage

You can run registrations using the command line interface. Configuration is handled via JSON files.

```bash
python transformation_graph.py --infile config.json
```

For detailed examples and tutorials, check out the [Examples Documentation](https://twardlab.github.io/emlddmm/build/html/examples.html).

## 📖 Documentation

Full documentation is available at [twardlab.github.io/emlddmm](https://twardlab.github.io/emlddmm/build/html/index.html).

It includes:
- [Installation Guide](https://twardlab.github.io/emlddmm/build/html/installation.html)
- [Output Specification](https://twardlab.github.io/emlddmm/build/html/output_specification.html)
- [API Reference](https://twardlab.github.io/emlddmm/build/html/index.html)

## 🤝 Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for more details.

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
