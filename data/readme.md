# Dataset and Experimental Results

## dataset
This directory contains the PDB IDs for the three datasets used in this study.

### Note on  Antigen–Antibody Dataset:

Two versions are provided.

- Antigen–Antibody_v1 (corresponds to the manuscript version posted April 06, 2026) : Contains complexes released between October 2023 and March 2024. To remove sequence redundancy, antibody and antigen sequences were clustered separately using MMseqs2 at sequence identity thresholds of 95% and 40%, respectively.This version contains 154 complexes.

- Antigen–Antibody_v2 (**corresponds to the latest revised version of the manuscript**) : Contains complexes released between January 2022, and March 2024. To remove sequence redundancy, antigen sequences were clustered using MMseqs2 at a 40% sequence identity threshold. This version contains 197 complexes.

**Key Difference:** v2 uses stricter deduplication, ensuring each complex represents a unique antigen cluster.

## Results
The results are organized according to the figure numbering, enabling direct correspondence with the figures.

This directory provides the evaluation results, including:
- DockQ scores for protein complexes.
- TM-scores for monomers.
