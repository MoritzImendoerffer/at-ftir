# Background

This (old) repo contains an approach to analyse data from refolding experiments measured via AT-FTIR. Due to the lack of proper analysis software like e.g. Simca and the urge to learn more about chemometrics, I implemented some algorithms by hand. The EMSC algorithm is implemented based on the book https://link.springer.com/book/10.1007/978-3-642-17841-2

The data consist of AT-FTIR spectra taken over time. Due to the high urea background which changes over time due to continuous dilution of the solution, the spectra are not trivial to analyse. Blank correction is applied based on reference spectra.

# Results

The raw spectra of the analysed samples:

![alt text](plots/pca/spectra)

# Disclaimer

This repo is quite old. The code is not very well structured but should work nevertheless.