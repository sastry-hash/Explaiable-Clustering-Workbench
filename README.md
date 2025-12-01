Explainable Clustering Workbench
--------------------------------

This project provides a simple, unified environment for running and explaining
multiple clustering algorithms using clear, interpretable rule-based explanations.

Supported algorithms:
- Threshold Trees (IMM-style)
- Explainable K-Means (Frost et al.)
- DBSCAN (explained using a surrogate decision tree)
- Spectral Clustering (explained using a surrogate decision tree)

The goal is to make clustering results easy to understand by generating:
- Decision-tree rules
- Surrogate explanations for black-box models
- Simple plots and metrics
- PoX (Price of Explainability) and fidelity comparisons

Project Layout:
---------------
src/
    1.py   - Threshold Tree implementation
    2.py   - Explainable K-Means
    3.py   - Surrogate explanations for DBSCAN & Spectral
    ui.py  - Simple Gradio interface

datasets/
    Cleaned versions of:
    - iris
    - wine
    - mall customers
    - wholesale customers

How to Run:
-----------
1. Install the required libraries:
   pip install numpy pandas scikit-learn gradio matplotlib scipy

2. To use the interactive UI:
   python src/ui.py

3. To run individual scripts:
   python src/1.py
   python src/2.py
   python src/3.py

Purpose:
--------
This workbench was created to make clustering models more transparent and easier
to interpret for research, learning, or practical use.

License:
--------
MIT License.
