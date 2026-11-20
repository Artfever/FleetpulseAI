# P23-S2: hosted explanation background and model disagreement

**Status:** local implementation and audit pass; a fresh hosted explanation check is still required after deployment.
**Issue:** [P23-S2 #174](https://github.com/Artfever/FleetpulseAI/issues/174)

## Deployment gap and fix

The 2026-11-21 hosted audit in `docs/p23_a3_remote_parity.md` showed that the
five-model raw comparison worked, but SHAP could not load the hosted
`data/raw/train_FD001.txt`. The serving explanation uses interventional SHAP
and needs a fixed reference sample. Shipping the full raw training history
would be unnecessary for inference, so this change packages only the
feature-only reference matrix used by that explainer.

The reproducible command is:

```powershell
.venv/Scripts/python.exe scripts/build_p23_shap_background.py
```

The output is `artifacts/final/explanation/decision-tree-background.json`:

- 100 rows × 96 ordered features, one per FD001/train engine at the fixed
  observed-cycle-90 cutoff (`fd001-train-all-engines-observed-cycle-90-v1`).
- Feature schema `past-only-v1`; no RUL labels or raw sensor histories are
  included. The file is not a model or trained weight bundle.
- Source training-history SHA-256:
  `963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8`, which
  matches the source hash pinned in the frozen Decision Tree manifest.
- Background content SHA-256:
  `1c7cf828d6bec51612e7752f28c70b440701b2eea4b4929bfc10217ea4211054`.
- Artifact payload SHA-256:
  `e1b0389f78b3399278f79332a199a212d92911b569e5599979518c9d87332073`.
- File size: 127,518 bytes.

The loader checks the artifact checksum, expected source digest, selection
policy, ordered schema, row count, finite values, and content digest. The
service prefers this packaged matrix; only a local checkout without it falls
back to building the same fixed background from the raw file. Synthetic tests
also verify that a deployed-style checkout can explain a prefix with the raw
training file absent.

## Real frozen-model audit

Command:

```powershell
.venv/Scripts/python.exe scripts/audit_p23_s2_explanation.py
```

No fit or refit was performed. For the packaged FD001/train engine-4 prefix
through cycle 164, the five frozen versions returned:

| Model | Version | Raw RUL (cycles) |
| --- | --- | ---: |
| Decision Tree | `decision-tree-rul-v1` | 29.794117647058822 |
| Random Forest | `random-forest-rul-v1` | 28.02408950947574 |
| XGBoost | `xgboost-rul-v1` | 26.14615821838379 |
| Constant mean baseline | `constant-rul-v1` | 107.80786195530997 |
| Dense RUL MLP | `dense-rul-mlp-final-v1` | 17.24176597595215 |

There were zero model failures. Raw max-minus-min disagreement was
90.56609597935783 cycles. All five identities and values agree with the
existing P23-A3 reference in `docs/p23_a3_remote_parity.md`.

The final Decision Tree explanation used version `decision-tree-rul-v1`,
manifest payload SHA-256
`ed434d11ce7ae22cd6445aa18fc4fb7e5db5f3438dded681b596ef54d6d37e89`, and
weight SHA-256
`c33021cb9cf5931aba545b6a327d0547f14be1363cbd0c24d57f0ad923dd7bb3`. It
returned raw RUL 29.794117647058822 cycles, baseline 126.93609647398786 cycles,
100 background rows, 96 feature contributions, and additivity error
`4.5322290098681606e-06` cycles. Example contributions: `sensor_4_mean`
−29.44410083770752 cycles and `sensor_13_mean` +6.585572566986084 cycles.
These are local model attributions for one simulated training prefix. They do
not identify causes or faults and are not accuracy, confidence, or maintenance
evidence.

## Verification and remaining check

- `.venv/Scripts/python.exe -m pytest -q tests/test_explanations.py` — PASS,
  6 tests, including checksum rejection and packaged-background inference
  without the raw history file.
- `.venv/Scripts/python.exe -m ruff check src/fleetpulse/services/explanations.py scripts/build_p23_shap_background.py tests/test_explanations.py` — PASS.
- `.venv/Scripts/python.exe scripts/build_p23_shap_background.py` — PASS; generated the verified 100 × 96 feature artifact above.
- `.venv/Scripts/python.exe scripts/audit_p23_s2_explanation.py` — local real bundle parity and SHAP check; exact output recorded above.

After the PR is merged and deployed, recheck the hosted RUL Prediction page:
select **Packaged FD001/train parity subset**, engine 4 at cycle 164, then run
**Show feature contributions** and **Compare frozen final models**. Confirm all
five versions and raw outputs, the final tree raw RUL and feature contributions,
and the displayed cycle units. The hosted commit SHA may not be exposed. Keep
#174 open until this remote result is checked; #262 depends on #174.

Full local verification on this branch:

- `.venv/Scripts/python.exe -m ruff check .` — PASS.
- `.venv/Scripts/python.exe -m pytest -q` — PASS, 734 tests.
- `.venv/Scripts/python.exe scripts/audit_p23_s2_explanation.py` — PASS; the
  real frozen-model outputs, SHAP additivity and raw prediction/explanation
  consistency check all match.
- `.venv/Scripts/python.exe -m streamlit run app/Home.py --server.headless true --server.port 18642` — started; `curl.exe --fail --silent --show-error http://localhost:18642/_stcore/health` returned `ok`; process stopped.
- `git diff --check` — PASS.
