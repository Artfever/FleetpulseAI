# P12-A3: Final tree model bundles

`python scripts/package_p12_final_tree_bundles.py` packages the verified P12-A1 refits into the ignored local directory `artifacts/final/tree/bundles/`. It refuses to overwrite a nonempty output directory. To make a changed bundle, choose a new versioned output directory and update the bundle constants and consumer explicitly.

The package contains separate manifests for the Decision Tree, Random Forest, selected depth-4 XGBoost, and the frozen constant-mean baseline, plus `final_tree_bundle_index.json`. Each manifest records its artifact role and version, ordered 96-feature past-only schema, frozen estimator configuration and seed, FD001/train provenance, source/selection/refit checksums, all 100 engine row counts, and checksums for serialized estimator weights. The mean baseline stores the selected statistic and training label count without a weights file.

Use `load_final_tree_bundle(path)` to validate the manifest checksum, expected identity/version, feature contract, training provenance, serialized model checksum, and estimator class before loading. `predict_final_tree_bundle(bundle, matrix)` requires a finite two-dimensional matrix in the saved feature order. Loading and prediction do not train or fit preprocessing. The final bundles use raw past-only engineered features; P12-A2 confirmed no learned tree preprocessing is needed.

Reproduction requires the ignored original FD001/train source and P12-A1 refit artifacts, plus tracked `artifacts/selection_manifest.json`. The local package is deliberately excluded from Git; code, tests, and this contract are tracked. Only FD001 training data and its complete-history targets are used. No official test data or labels are used in packaging.

Verification performed for this task: package and reload each of the four local bundles, check repeated predictions for the training reference row against the P12-A1 saved outputs, inspect every manifest/index checksum, and run the focused and full test suites. The training-row values are serialization/inference checks, not evaluation metrics.
