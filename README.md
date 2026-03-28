# ML Challenge: Random Forest in `pred.py`, evaluation in `eval_pred.py`

## `pred.py`
- Purpose: Random Forest inference on a CSV — one painting prediction per row.
- Main function: `predict_all(filename)`
- Allowed imports only: `numpy`, `pandas`, `sys`
- Model file used: `rf_model_params.npz`
- Behavior:
  - Reads raw/dirty survey rows.
  - Applies built-in cleaning + feature engineering.
  - Runs Random Forest inference from exported parameters.
  - Returns labels:
    - `The Persistence of Memory`
    - `The Starry Night`
    - `The Water Lily Pond`


## `eval_pred.py` (local evaluation helper)
- Purpose: Evaluate current `pred.py` model on an 80/20 split and print:
  - Training accuracy + confusion matrix
  - Validation accuracy + confusion matrix
  - Training and Validation sets do not overlap
- Uses:
  - `ml_challenge_dataset.csv` for full labeled data
  - `holdout_20_info.csv` if present (fixed validation indices)
  - Otherwise creates a stratified 80/20 split

