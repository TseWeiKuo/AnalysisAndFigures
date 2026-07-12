"""Configuration helpers shared by tracking-QC-aware plotting workflows."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrackingQCConfig:
    """Tracking-QC options without coupling them to a plotting workflow."""

    enabled: bool = False
    error_thresholds: dict | None = None
    min_cameras: int = 2
    max_interp_gap_frames: int = 4
    min_valid_fraction: float = 0.8

    def output_metadata(self):
        """Return the common QC fields written to result dataframes."""
        return {
            "Apply_Tracking_QC": bool(self.enabled),
            "Min_Cameras": self.min_cameras if self.enabled else np.nan,
            "Max_Interp_Gap_Frames": (
                self.max_interp_gap_frames if self.enabled else np.nan
            ),
            "Min_Valid_Fraction": (
                self.min_valid_fraction if self.enabled else np.nan
            ),
        }


def from_legacy_arguments(
        apply_tracking_qc=False,
        tracking_error_thresholds=None,
        min_cameras=2,
        max_interp_gap_frames=4,
        min_valid_fraction=0.8
):
    """Build a config while public plotting methods retain existing arguments."""
    return TrackingQCConfig(
        enabled=apply_tracking_qc,
        error_thresholds=tracking_error_thresholds,
        min_cameras=min_cameras,
        max_interp_gap_frames=max_interp_gap_frames,
        min_valid_fraction=min_valid_fraction,
    )
