-# Citation Network Analysis Tools

A comprehensive suite of Python tools for analyzing academic citation networks, including knowledge propagation patterns, influence metrics, and network visualization.

## Overview

This project provides tools for analyzing citation networks to understand influence propagation patterns, identify key papers, and evaluate knowledge transmission in academic literature. It includes multiple analysis models and visualization capabilities.

## Features

- **Citation Network Building**
    - Build citation networks from various data sources (OpenAlex, arXiv)
    - Process paper metadata and section information
    - Handle external references and URIs
    - Support for multiple export formats

- **Analysis Models**
    - SAIS (Susceptible-Alert-Infected-Susceptible) model for information spread
    - Forward Scholarly Propagation Centrality (FSPC) analysis
    - PageRank-based influence analysis
    - Sensitivity and specificity metrics
    - Economic impact assessment

- **Visualization Tools**
    - Network structure visualization
    - Influence propagation patterns
    - Time evolution curves
    - Parameter sensitivity heatmaps
    - Impact distribution plots

## Requirements

- Python 3.11.12
- Virtual environment (virtualenv)

### Dependencies

- networkx~=2.8 
- numpy~=2.2.3 
- requests~=2.32.3 
- matplotlib~=3.10.1 
- scipy~=1.15.2 
- urllib3~=2.2.2 
- seaborn~=0.13.2 
- pandas~=2.2.3 
- tqdm~=4.67.1 
- arxiv~=2.2.0 
- lxml

## Installation

1. Clone the repository:

2. Create and activate a virtual environment:


````

python -m venv venv source venv/bin/activate # On Windows: venv\Scripts\activate
````



3. Install dependencies:

````

pip install -r requirements.txt
````


## Usage

### Building Citation Networks

````

python network.py [input_directory] [output_path] --format [gexf|graphml|edgelist|adjlist|pickle]
````


### Running SAIS Model Analysis

````

python sais.py path/to/graph.pkl -s [number_of_steps]
````


### Performing PageRank Analysis

````

python pagerank.py path/to/graph.pkl
````


### Analyzing Economic Impact

````

python economic.py [network_path] --max-depth [depth]
````


## Key Components

1. **Network Building (`network.py`)**
    - Creates citation networks from academic paper data
    - Supports multiple export formats
    - Handles paper metadata and references

2. **SAIS Model (`sais.py`)**
    - Simulates information spread in networks
    - Parameter optimization
    - Multiple visualization methods

3. **PageRank Analysis (`pagerank.py`)**
    - Standard and modified PageRank calculations
    - Super-propagator detection
    - Impact visualization

4. **Economic Analysis (`economic.py`)**
    - Multiple economic models
    - Innovation loss calculation
    - Knowledge productivity assessment

5. **OpenAlex Integration (`openalex.py`)**
    - Fetches citation data from OpenAlex API
    - Builds citation networks
    - Handles rate limiting and API errors

## Data Format

Citation networks are stored in NetworkX graph format with the following node attributes:

- `title`: Paper title
- `year`: Publication year
- `sections`: List of paper sections containing:
    - `section_name`: Section name
    - `external_uris`: External references
    - `cited_references`: Citation information

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the GNU General Public License v3.0 (GPLv3).

## Citation

If you use this tool in your research, please cite:

````

Chamorro Padial, Jorge; Rodrigo-Ginés, Francisco-Javier; Rodríguez Sánchez, Rosa María; Gil Iranzo, Rosa Maria; García González, Roberto, 2025, "Scripts for: The economics of lost knowledge: modeling the knowledge cost due to non-FAIR data practices", https://doi.org/10.34810/data2383, CORA.Repositori de Dades de Recerca
````
