# P11-S2: bounded MLP architecture comparison

## Predeclared protocol

Compared three hidden-layer widths: `(32, 16)`, the existing P10 baseline
`(64, 32)`, and `(128, 64)`. All other settings stayed fixed: 30-cycle,
96-feature `past-only-v1` inputs; seed 42; batch size 128; Adam learning rate
0.001; MSE loss; maximum 20 epochs; patience 3; `min_delta=0`; CPU execution.
The same 80 fitting engines and 20 saved FD001/train validation prefixes were
used for each run. Each run's best checkpoint is selected using validation
MSE, as in the existing P10 procedure.

Reproduce with:

```powershell
.venv/Scripts/python.exe scripts/run_p11_s2_mlp_settings.py
```

The script saves a versioned weights/manifest bundle and per-engine results in
separate ignored folders under `artifacts/experiments/P11_S2_mlp_settings/`.
Each saved bundle is checked in a fresh process with
`scripts/verify_mlp_artifact.py`. The existing P10 bundle is not overwritten;
its weights remain SHA-256
`a5dcf637a1178f44c837688646295250971dca5862cd934e3935293be53e6273`.

## Results

Python 3.12.10, PyTorch 2.15.0+cpu, scikit-learn 1.9.1. Best epoch is the
minimum validation MSE among epochs actually trained.

| Hidden layers | Best epoch / epochs trained | Validation MSE (cycles²) | MAE (cycles) | RMSE (cycles) | Runtime (s) | Fresh reload |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `(32, 16)` | 17 / 20 | 532.509961 | 15.269108 | 23.076177 | 3.505 | 20/20 exact |
| `(64, 32)` | 12 / 15 | 526.151562 | 15.511258 | 22.937993 | 2.748 | 20/20 exact |
| `(128, 64)` | 8 / 11 | **486.915088** | **15.080604** | **22.066152** | 2.083 | 20/20 exact |

Every run used seed 42. The larger network had the lowest selected validation
MSE, MAE, and RMSE in this three-setting comparison. It also stopped at epoch
8 after 11 epochs were trained under patience 3. That result is a candidate
for the P11 discussion, not a final architecture decision.

## Interpretation and limits

The same validation split selected the epoch in each run and is also being
used to compare architecture widths. These are development-selection results,
not an independent estimate of generalization. The per-run validation MSE is
selection evidence. Runtime is machine-specific and not a reliable speed
comparison across environments. No official test histories or RUL answers
were opened.

The baseline `(64, 32)` rerun matched the P10 saved selection: epoch 12 and
validation MSE 526.1515625 cycles². The original P10 weights hash remained
unchanged. The pinned source SHA-256 is
`963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8`; semantic
split-manifest SHA-256 is
`6fd1d32f7570f70f5a9513d900fbb9d26b182359aced425e2276700c84f08d92`.
