# P15 fleet prediction contract

`fleetpulse.services.extract_latest_engine_histories` is the P15-A1 input
boundary. One call handles one dataset partition. Each result retains the
dataset-qualified identity `(dataset_partition, unit_id)` so engine 1 in one
partition cannot be confused with engine 1 in another. Callers must not combine
different dataset partitions in one input batch.

The service accepts normalized raw-history mappings. It groups interleaved rows
by engine while preserving their order within each engine, then validates each
engine independently with the shared quality contract. Valid entries contain
the complete observed prefix through its latest cycle. Cycle gaps remain
warnings; invalid columns, readings, duplicate cycles, or cycle ordering errors
are returned as per-engine errors. A bad engine does not remove or mislabel a
valid peer. An empty batch, unidentifiable row, or invalid partition is rejected
as a batch-level input error because no safe dataset-qualified result identity
can be attached.

P15-A1 prepares validated histories only; it does not load a model or predict.
`fleetpulse.services.predict_fleet` consumes the prepared histories and invokes
the shared saved-model predictor for each valid engine. It returns one
`FleetPredictionResult` per identified engine, including observed cycle, raw
and displayed RUL cycles, model/version, feature-window rows, warnings, and
error fields. A history or prediction failure for one engine remains on that
engine's result while valid peers retain their predictions. Batch-level rows
without an identifiable unit ID still fail the request because no safe result
identity can be preserved. Sorting and filtering belong to later UI tasks and
must retain these result identities.

## Interpreting planning priority

A lower predicted remaining useful life (RUL) can place an engine earlier in a
planning review. It is a model estimate from the observed history prefix, in
operating cycles. It is not a confirmed physical fault, a fault probability, a
sensor alarm, or a guarantee that an engine will last the estimated number of
cycles. A prediction alone does not diagnose a cause or authorize maintenance.

Treat the ordering as a decision-support cue for further review. Check the
underlying readings and their quality, relevant maintenance records, and the
operating context; have a qualified person decide what action is appropriate.
An invalid history or failed prediction has no RUL estimate and must not be
silently ranked as though it had a valid value. Equal estimates remain separate
engines, identified by their dataset partition and unit ID. These limits are
especially important for this development model: the available parity examples
use short prefixes of FD001 training histories and do not establish official
test performance or maintenance readiness.

For example, the local `decision-tree-rul-v1` parity run on packaged
`FD001/train` prefixes observed:

| Engine | Observed cycle | Predicted RUL (cycles) | Input rows | Note |
| ---: | ---: | ---: | ---: | --- |
| 4 | 164 | 34.19607843137255 | 30 | No short-window warning |
| 5 | 58 | 158.16 | 30 | No short-window warning |
| 12 | 7 | 184.35714285714286 | 7 | `short_window` warning |

These values are example model outputs on training-history prefixes, not
official-test answers or observed failure times. Their differences illustrate
that predicted RUL is not itself a diagnosis; even a valid number may come from
a short input window and still needs context before anyone acts on it.
