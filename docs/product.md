# FleetPulse AI user guide

FleetPulse estimates remaining useful life (RUL) in operating cycles from simulated NASA C-MAPSS FD001 engine histories. Maintenance Planner uses those estimates or example jobs to compare schedules with explicit simulation assumptions.

Start the app from the repository root with `python -m streamlit run app/Home.py`. Use the sidebar or the Home demo links.

| Screen | What to demonstrate |
| --- | --- |
| **Home** | Rotate the aircraft preview and launch Engine Replay or the scheduling sandbox. The readiness panel shows whether the packaged telemetry and the saved replay estimate are available. |
| **Fleet Overview** | Inspect the lowest visible estimates in the 3D fleet deck. Set a review threshold, select engines and save them for the Planner. The threshold display filter leaves the saved selection intact. |
| **Engine Explorer** | Orbit the aircraft, inspect the turbofan cutaway, and play Engine 4's recorded history. Open the telemetry and evidence section for observed sensors, training labels and past-only features. |
| **RUL Prediction** | Select a history and model or validate an FD001-formatted CSV. The 3D trace displays saved-model results at evaluated observed cutoffs. A CSV prediction uses the latest observed cycle and can be sent to the Planner. |
| **Maintenance Planner** | Play a schedule in the 3D hangar and switch between EDD and seeded GA at the same cycle. Build a saved fleet scenario or use the sandbox to compare a change in assumptions. |
| **Model Laboratory** | Switch between MAE and RMSE in the 3D arena. Engine Scan shows signed errors from verified official FD001/test results. Development validation, studies and downloads have their own evidence sections. |

The 3D aircraft, turbofan and hangar geometry are illustrations. Estimates, checkpoint positions, schedule assignments and evaluation scores come from their labeled data sources. Missing or invalid data and artifacts produce an unavailable state. Use the [data setup guide](data_policy.md) and [live demo runbook](p25_live_demo_runbook.md) to prepare the local demonstration.
