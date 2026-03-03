# MAI643 Project

This repository contains course project materials and deliverables for MAI643.

## Run Deliverable 2 Notebook

Notebook path:
- `deliverable2/Deliverable2_Notebook.ipynb`

Dataset path used by the notebook:
- `deliverable2/data/healthcare-dataset-stroke-data.csv`

### 1. Get the project files

Option A (recommended): clone the full repository.

```bash
git clone https://github.com/Larkoss/MAI643-Project
cd MAI643-Project
```

### 2. Create a virtual environment

From the project root (`MAI643-Project`):

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install jupyter numpy pandas matplotlib seaborn scikit-learn imbalanced-learn
```

### 4. Start Jupyter and open the notebook

```bash
jupyter notebook
```

Then open:
- `deliverable2/Deliverable2_Notebook.ipynb`

### 5. Run the notebook

1. In Jupyter, select `Kernel -> Restart & Run All`.
2. Wait for all cells to complete.
3. Generated outputs will be written under:
   - `deliverable2/figures/`
   - `deliverable2/tables/`
