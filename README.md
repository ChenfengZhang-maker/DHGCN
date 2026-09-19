# DHGCN

Official PyTorch implementation of the DHGCN model for drug repositioning.

## Requirements

To run this code, please ensure you have the following environments installed:
- Python >= 3.8
- PyTorch >= 1.12.0
- torch_geometric
- pandas
- numpy
- scikit-learn
- matplotlib

## Repository Structure

- `main.py`: The complete DHGCN framework. Run this script for the full model training and evaluation.
- `data/`: Directory containing the processed DGD dataset splits (train, validation, and test).

## Usage

To train and evaluate the full model, simply execute:

```bash
python main.py
