# Overview

This repository provides an approach for analyzing data from refolding experiments gathered via Attenuated Total Reflectance Fourier Transform Infrared Spectroscopy (AT-FTIR). The experimental process involved:

- Dissolving inclusion bodies using a high urea concentration.
- Gradually diluting the solution into a refolding buffer.
- Measuring AT-FTIR spectra at distinct intervals.

The data comprises AT-FTIR spectra captured over time. Owing to the high urea background that changes over time, the spectra analysis is complex. The feasibility of blank correction was assessed using reference spectra, but was not employed in this analysis.

Due to the absence of specialized analysis software like Simca, and to further chemometrics understanding, certain algorithms were manually implemented. The Extended Multiplicative Signal Correction (EMSC) algorithm, in particular, was implemented based on [this reference book](https://link.springer.com/book/10.1007/978-3-642-17841-2).

## Disclaimer
Please note, this repository is dated and the code, though functional, lacks structural clarity.

# Objective
The primary objective was to determine correlations between experiments and enhance understanding of why certain experiments were unsuccessful. The exploratory analysis groups experiments based on their processed AT-FTIR spectra over time, with no intent to regress inputs with outputs.

# Methodology

The analysis involves the following steps:

- Gathering raw data from Excel files into a [consolidated dataframe](export/one_to_rule_them_all.xlsx) using the [get_data script](scripts/get_data.py).
- Preprocessing and clustering the data with the [AtFtirAnalysis class](scripts/include/AtFtirAnalysis.py), which primarily slices relevant wavenumber regions and performs baseline correction if necessary. The classification employs Principal Component Analysis (PCA) and the correlation matrix of each spectra, converted into an adjacency matrix. This process is detailed in the [graph_analysis notebook](scripts/graph_analysis.py).
- The complete analysis is presented in the [data_analysis notebook](scripts/data_analysis.ipynb).

# Results

- [Raw Spectra](plots/pca/spectra.png)
- [Preprocessed and Analyzed Spectra](plots/pca/analyzed_spectra.png)

The subsequent plot presents the evolution of the principal components for each spectra over time, revealing four clusters: the blank runs, one outlier (a failed experiment 171017), and two subclusters for the remaining runs.

- [Cluster Analysis at Equilibrium](plots/clustermaps)
- [PCA of Spectra over Time](plots/pca/pca.png)

The cluster analysis and correlation matrix yield similar results for all runs except the failed experiment (171017).

- [Cluster at Equilibrium](plots/clustermaps/clustermap_t60_1200_1800.png)

The correlation matrix can be transformed into an adjacency matrix and further into a graph, though this requires a threshold to determine connections between experiments.

- [Graph at Equilibrium](plots/igraph/igraph.png)

# Discussion

In essence, AT-FTIR is a viable tool for monitoring chemical reactions over time. However, the presence of high background noise, especially varying urea concentration, presents significant challenges. It's possible to subtract spectra from blank or control runs displaying only background signals, but this may introduce unwanted artifacts. For this analysis, blank runs weren't subtracted, as all samples underwent similar processing. This allowed for basic clustering. The analysis suggests potential for identifying incorrect starting conditions and variations among runs. More trials are needed to deduce significant conclusions regarding principal component variations for the refolding reaction.