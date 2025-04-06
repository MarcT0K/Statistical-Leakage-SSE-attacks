# Experiments of the paper "Revisiting the attacker's knowledge in inference attacks against Searchable Symmetric Encryption"

Link to the paper: https://eprint.iacr.org/2023/1883

Authors: Marc Damie, Jean-Benoist Leger, Florian Hahn, and Andreas Peter.

For **any question** about the code or the paper, contact Marc Damie.

## Install

The install process is straightforward: `bash setup.sh`. This script installs the Python dependencies and download the datasets.

## Reproduce

To repoduce our results, you need to run `python3 generate_results.py`. This script launches all the experiments one by one. The results will be stored in multiple CSV files in a `results` folder.

To generate the figures, you need to run `python3 generate_figures.py`. This script generates all figures one by one using the CSV files generated by the previous script.

## Repository structure

The folder `src/` contains the following elements:

- `document_extraction.py` contains the functions to process the datasets. Each dataset has a different format so there are dedicated functions for each datasets.
- `keyword_extraction.py` contains the functions to process the extracted documents. These functions use multiprocessing to extract keywords efficiently.
- `simulation_utils.py` contains all auxiliary functions necessary to simulate attacks; e.g., adversary knowledge generation.
- `attacks/` contains the functions to perform the score and IHOP attacks.
