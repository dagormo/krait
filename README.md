Krait is an experimental platform for ion chromatography method development. It combines:
  Python engine (engine/) for descriptor calculation, PCA projection, retention time prediction, and gradient simulation.
  Clients for user interaction:
    Streamlit app (clients/predict/) → UI for single-/multi-analyte prediction and gradient simulation.
    Chromeleon SDK plugin (clients/chromeleon_plugin/) → C# integration with Thermo Fisher Chromeleon using the SDK.

Repository structure
MordredVersion/
  engine/                  # Core Python engine
    descriptors.py         # SMILES → Mordred/PaDEL descriptors
    pca.py                 # PCA projection
    predictor.py           # Feature vector assembly + prediction
    preprocess.py          # Preprocess analyte descriptors once
    simulation.py          # Gradient calibration + prediction
    config.py              # Central paths (resources, models)

  clients/
    predict/               # Streamlit app (Python)
      test_UI_multistep.py
    chromeleon_plugin/     # Chromeleon SDK plugin (C#)
      CmSdkPoC.sln
      CmSdkPoC/
        Chromeleon/
        UI/
        Utils/
        Program.cs
        ...

  data/                    # Resources (descriptor template, PaDEL jar)
  pkl/                     # Trained model files (.pkl)
  tests/                   # Unit/integration tests

  requirements.txt         # Python dependencies
  README.md                # This file

Python: Engine + Streamlit app
  1. Setup
    # Create and activate a venv
    python -m venv .venv
    source .venv\Scripts\activate.ps1
    
    Install dependencies
    pip install -r requirements.txt

  2. Run the Streamlit app
    cd MordredVersion
    streamlit run clients/predict/test_UI_multistep.py
    Streamlit will open at http://localhost:8501
    You can paste analytes (Name + SMILES) and define gradient conditions to predict retention times and simulate chromatograms.
    Chromeleon SDK plugin (C#): Location: clients/chromeleon_plugin/. Open CmSdkPoC.sln in Visual Studio. The project contains helpers for reading/writing Chromeleon method files and integrating with the Krait prediction service.

Build
  Open in Visual Studio (2019 or later recommended).
  Build the solution (Ctrl+Shift+B).  
  Output assemblies will appear under bin/ (ignored by Git).

Development notes
  Resources:
    Descriptor template: data/descriptors_noredundancy.csv
    PaDEL JAR: data/PaDEL-Descriptor/PaDEL-Descriptor.jar
    Trained models: pkl/logk_model_hydroxide.pkl, pkl/pca_model_95pct.pkl

  Design:
    Heavy descriptor calculations are run once per analyte (preprocess.py).
    Multiple condition predictions reuse preprocessed features (predictor.py).
    Streamlit and Chromeleon plugin both consume the same Python engine for consistency
