# C-MAPSS raw and demo data

## Training and test histories

Git includes the original NASA C-MAPSS histories in `data/raw/`.
Each of FD001, FD002, FD003, and FD004 has a `train_*.txt` and `test_*.txt`.
These files preserve the source bytes. FD001 hashes are recorded in the
[provenance record](fd001_provenance.md). FD002 through FD004 hashes are
recorded in the manifests and evaluation reports under `artifacts/environments/`.

The app uses the training histories as observed inputs. Test histories are
used for offline evaluation. The separate `RUL_FD*.txt` answer files remain
Git-ignored and are supplied locally for offline scoring. Obtain them from the
[NASA dataset archive](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data).
Extract the corresponding files from `CMAPSSData.zip` and preserve their bytes.

## Demo and test data

The explicitly selected `Illustrative synthetic example` on the prediction page reads `data/demo/illustrative_history.txt`. It contains the three artificial rows of the existing project-created fixture `tests/fixtures/synthetic_engine_history.csv`, converted to the existing whitespace format. These invented numbers are not NASA observations, are not a model-training set, and are not representative engine readings. Their purpose is to exercise the interface. A saved trained bundle must still be supplied separately, and predictions on these out-of-distribution values cannot establish model quality or the Phase 7 real-data parity gate. The UI labels this distinction and uses `illustrative/synthetic` as its partition.

The explicitly selected `Packaged FD001/train parity subset` reads `data/demo/fd001_training_prefixes.txt`. It contains only original simulated training readings from engine 4 cycles 1–164, engine 5 cycles 1–58, and engine 12 cycles 1–7 (229 rows total). The owner approved including this small subset for the public deployment demonstration on 13 November 2026. The packaging script checks the original `train_FD001.txt` hash, preserves selected source rows, and rejects missing cycles or any unexpected source. The subset is not a complete dataset, validation split, test partition, or evidence of model quality. It includes no official test histories or RUL answers.

The NASA [PCoE source page](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) identifies and distributes the dataset but does not state a dataset-specific redistribution license.
