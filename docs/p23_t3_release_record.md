# P23-T3 release URL, version, and fixes record

## Release address and reported deployment

- Streamlit URL: <https://fleetpulseai.streamlit.app/>.
- Abdullah reports that he opened the URL and saw the application deployed and
  running. The project had previously reported that the app deploys from
  `dev`; at the time of that owner check, `dev` was `b296e5e8c2bd4c391c864bb0afa92c71b2c4bd75`.
- The hosting platform does not expose its deployed SHA through this repository,
  and no deployment log was available here. Do not claim that `b296e5e` is the
  exact hosted SHA. Current `dev` is later and includes the P23-S1/T2 audit
  merges; whether those later commits have redeployed is not independently
  confirmed.
- Anonymous requests from this environment currently receive HTTP 303 to
  Streamlit sign-in for both `/` and `/_stcore/health`. Abdullah's authenticated
  browser check differs; remote behavior is therefore owner-reported, not
  independently reproduced by this audit.

## Runtime and frozen model versions

Repository runtime declarations: Python `>=3.12,<3.13`, Streamlit `1.67.0`,
package version `0.1.0`, entrypoint `app/Home.py`. The five tracked frozen final
model versions are Decision Tree `decision-tree-rul-v1`, Random Forest
`random-forest-rul-v1`, XGBoost `xgboost-rul-v1`, baseline `constant-rul-v1`,
and Dense MLP `dense-rul-mlp-final-v1`.

Command: `.venv/Scripts/python.exe scripts/smoke_p23_a1_release.py`.

Observed inference-only output on packaged simulated FD001/train engine 4,
164 rows through cycle 164, with no source training file and no fitting:

| Model version | Raw RUL (cycles) |
| --- | ---: |
| `decision-tree-rul-v1` | 29.794117647058822 |
| `random-forest-rul-v1` | 28.02408950947574 |
| `xgboost-rul-v1` | 26.14615821838379 |
| `constant-rul-v1` | 107.80786195530997 |
| `dense-rul-mlp-final-v1` | 17.24176597595215 |

All five bundles loaded with zero failures. Demo-history SHA-256:
`953489b722a33815ee08f360c1a677d314f318b7d74d8bef88cd31dde0a8f9a7`. These
are smoke estimates from one simulated training history, not accuracy or
maintenance evidence.

## Fixes recorded in merged P23 work

- PR #508 packaged the five final serving bundles and pinned the optional
  XGBoost/MLP dependencies.
- PR #510 added a clean-checkout inference audit using only tracked demo input.
- PR #512 enforced the existing 1 MiB CSV limit at the Streamlit upload widget.
- PR #514 fixed Windows CRLF handling in the frozen-manifest audit and verified
  the Model Laboratory's local metric rows against the P13 report/bundles.
- PR #515 recorded local cold-start and full-flow evidence; a fresh local app
  returned health `ok` in 3.395s and 4.612s.

No model was retrained on request. No other remote failures or fixes are claimed:
anonymous authentication redirects prevent inspection of hosted logs and screen
errors in this environment. The P13 report and development-model artifacts are
not tracked in `dev`, so their hosted availability needs explicit confirmation.

## Verification

- `.venv/Scripts/python.exe scripts/smoke_p23_a1_release.py` — PASS; five
  final models loaded, zero failures, inference only.
- `curl.exe --silent --show-error --max-redirs 0 --output NUL --write-out "anonymous app HTTP %{http_code} redirect=%{redirect_url}" https://fleetpulseai.streamlit.app/` — HTTP 303 to Streamlit sign-in.
- Same request to `https://fleetpulseai.streamlit.app/_stcore/health` — HTTP
  303 to Streamlit sign-in.
- `.venv/Scripts/python.exe -m ruff check .` — prior full PASS on this `dev`
  code base; this T3 change adds documentation only.

The URL and run status are owner-reported. Exact deployed SHA, fresh remote
version screen, and hosting logs remain unverified here.
