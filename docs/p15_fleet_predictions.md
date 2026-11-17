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
P15-A2 consumes the returned identities, prefixes and per-engine errors to run
batch inference and attach estimates, cycles, and model versions. Sorting and
filtering belong to later UI tasks and must retain these identities.
