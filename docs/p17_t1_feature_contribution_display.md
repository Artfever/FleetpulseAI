# P17-T1: signed feature contributions in the RUL page

The RUL page now offers **Show feature contributions** for a selected observed
history. It displays the frozen final Decision Tree's raw RUL output separately
from the development prediction, then shows the five largest positive and five
largest negative SHAP contributions. Each row names the feature, its input
value, and signed contribution in cycles. Positive values move the raw model
output above the reference baseline; negative values move it below. The page
also displays the baseline and additivity error.

This is a local model attribution relative to the documented cycle-90
FD001/train reference sample. It is not a causal effect, confidence interval,
accuracy score, or maintenance recommendation. The RUL page reports a clear
unavailable message if the frozen bundle, raw training source, or SHAP runtime
is missing or incompatible. SHAP 0.54.1 is now a core pinned dependency so the
interactive feature works after the normal project install.

## Verification

Synthetic AppTest coverage in `tests/test_prediction_page.py` checks that the
selected observed prefix is passed to the final-tree explanation service and
that positive and negative rows are shown in cycles. The full local frozen
bundle UI audit is:

```powershell
.venv/Scripts/python.exe scripts/check_p17_t1_contribution_display.py
```

The audit uses packaged FD001/train engine 4 through cycle 164 and writes its
full output to the ignored `reports/P17_T1_contribution_display.json`. The
local model/data bundles are not committed.

Actual local output on 18 November 2026: frozen model `decision-tree-rul-v1`,
raw RUL **29.7941176471 cycles**, reference baseline **126.9360964740
cycles**, and additivity difference **0.0000045322 cycles**. The UI's top
positive rows began with `sensor_13_mean` (+6.5856 cycles), `sensor_17_std`
(+2.6715), and `sensor_11_std` (+0.7105). Its top negative rows began with
`sensor_4_mean` (−29.4441 cycles), `sensor_9_mean` (−13.3104), and
`sensor_21_mean` (−11.7710). The numbers are model allocations relative to
the selected training background; the signs indicate movement relative to
the baseline, not causal sensor effects.
