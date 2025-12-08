# Multimodal Deep Learning Approach for Cyclic Peptide-Based Antibiotic Detection

Antibiotic resistance is a growing global concern, making it essential to develop fast and effective ways to detect antibiotics. Cyclic peptides (CPs) are promising candidates for biosensing due to their stability, specificity and ability to form complex structures. This project focuses on using deep learning to design CPs for antibiotic detection, combining multiple data sources to create a powerful AI-driven model. A key component of our approach involves using Molecular Dynamics simulations to generate interaction data between CPs and antibiotics, capturing their behavior under different conditions.


## Overview of added files

- **Contact_Maps_MD.pdf** – contact maps from 200 ns MD simulations showing, for each residue of the cyclic peptide, how frequently it forms hydrogen-bond contacts with the antibiotic over time.

- **HBonds_MD.pdf** – time-resolved plot of the number of hydrogen bonds between the cyclic peptide and the ligand across 200 ns; the solid line shows the average over replicas, and the shaded region represents the standard deviation.

- **LIE_MD.pdf** – Linear Interaction Energy (LIE) analysis: a time series of estimated ΔΔG (binding free-energy change) for the peptide-ligand complex, with the mean value and its standard deviation along the trajectory.

- **MD_Sim_Script.ipynb** – main notebook used to set up, run and post-process MD simulations (system preparation, equilibration/production runs, trajectory loading, and calculation of observables such as hydrogen bonds).

- **Min_Dist_MD.pdf** – plot of the minimum distance between the cyclic peptide and the ligand as a function of simulation time, including the average curve and its standard deviation, used to monitor binding and unbinding events.

- **Parser_PDB.ipynb** – notebook for parsing PDB files to build the dataset of peptide-ligand complexes: it extracts protein–ligand pairs, identifies binding-site residues within 4 Å of the ligand, converts them to three-letter sequences with and without amino acid coordinates and aggregates these data into tables for further analysis.

- **Peptide_Checker.ipynb** – analysis notebook that compares LLM-generated cyclic peptides with the training set (e.g. sequence similarity, amino-acid statistics and distributional shifts) to assess how novel or close to the data the generated sequences are.

- **Prompt_Chem_LLM.py** – script for running the ChemLLM-7B model with a custom peptide–ligand completion prompt, feeding in the curated corpus and generating new cyclic peptide candidates.

- **cPep_time_csv.csv** – initial database of cyclic peptide–antibiotic complexes and MD-derived features, used as the starting point for building the peptide–ligand dataset.
