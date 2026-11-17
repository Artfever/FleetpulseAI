# P12-A2: final preprocessing audit

The final Decision Tree, Random Forest, and XGBoost models use the same frozen
96-column `past-only-v1` feature schema over a 30-cycle window. Feature
engineering computes fixed prefix statistics; it fits no parameters to the
data. The tree estimators and constant baseline consume those values directly,
so adding a scaler would change the frozen P11 input contract without a model
selection decision to support it.

The MLP is the only P12 model with learned preprocessing. Its P12-S1
`StandardScaler` is already fit on all 20,631 original FD001/train feature rows
from 100 engines. P12-S3 packages those exact means, scales, and variances with
the final MLP. No official test histories or labels are involved.

## Verify

After P12-A1/#63 and P12-S1/#64 outputs are available locally, run:

```powershell
.venv/Scripts/python.exe scripts/audit_p12_preprocessing.py
```

The audit checks the P12-A1 refit manifest and file hashes, the unchanged P11
parameters/feature configuration, the mean baseline's use of all training
labels, and the P12-S1 scaler source, configuration, and row counts. It
confirms there is no learned preprocessing state attached to the tree models
and that the MLP's scaler uses the complete original training source.
