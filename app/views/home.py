"""FleetPulse AI view."""

from fleetpulse.ui import render_page, render_training_data

render_page(
    "FleetPulse AI",
    "Simulated turbofan maintenance planning.",
    "Verified model bundles and evaluation reports appear only when matching artifacts are installed.",
    "Open Model Laboratory to see which verified results are available in this deployment.",
)
render_training_data()
