# ML models for Higgs/Recoil Identification from the Higgs Boson to four quarks decay 

## Data Extraction
extract.py extracts the data from ROOT files and converts them to numpy arrays stored in a single hdf5 file.
Data is then normalized using normalize.py.

## Neural Networks
Both NNs use the Tensorflow's keras library.
The GNN has a combination of input, output, conv1D and dense layers.
The DNN uses only input, output, and dense layers.
