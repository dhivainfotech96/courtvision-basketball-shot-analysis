"""
shot_comparator.py
CourtVision - Multi-Shot Comparison Module

Compares two basketball shots:
1. Trajectory arc overlay plot
2. Side-by-side metric comparison table
3. Image-plane kinematic insights (arc steepness, hoop proximity, duration)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Tuple
from ball_tracker import BallDetection
from trajectory_analyzer import ShotMetrics


class ShotComparator:
    """
    Compares two shot trajectories and generates comparative visualisations
    and metrics reports.
    """

    @staticmethod
    def build_comparison_dataframe(
        name_a: str, metrics_a: ShotMetrics,
        name_b: str, metrics_b: ShotMetrics
    ) -> pd.DataFrame:
        """
        Builds a tabular comparison of two shots with image-plane metrics.
        """
        data = {
            "Metric (Image Plane)": [
                "Launch Angle (deg)",
                "Closest Distance to Hoop (px)",
                "Apex Coordinates (X, Y px)",
                "Release Point (X, Y px)",
                "Flight Duration (s)",
                "Tracked Frames Count",
                "Tracking Success Rate (%)",
                "Average Image Speed (px/s)",
            ],
            f"Shot A ({name_a})": [
                f"{metrics_a.launch_angle_deg:.1f}°" if metrics_a.launch_angle_deg is not None else "N/A",
                f"{metrics_a.closest_distance_px:.1f} px" if metrics_a.closest_distance_px is not None else "N/A",
                f"({metrics_a.apex_point[0]}, {metrics_a.apex_point[1]})" if metrics_a.apex_point else "N/A",
                f"({metrics_a.release_point[0]}, {metrics_a.release_point[1]})" if metrics_a.release_point else "N/A",
                f"{metrics_a.flight_duration_sec:.2f} s",
                f"{metrics_a.detected_frames_count} / {metrics_a.total_frames_analyzed}",
                f"{metrics_a.tracking_success_rate:.1f}%",
                f"{metrics_a.avg_speed_px_per_sec:.1f} px/s" if metrics_a.avg_speed_px_per_sec is not None else "N/A",
            ],
            f"Shot B ({name_b})": [
                f"{metrics_b.launch_angle_deg:.1f}°" if metrics_b.launch_angle_deg is not None else "N/A",
                f"{metrics_b.closest_distance_px:.1f} px" if metrics_b.closest_distance_px is not None else "N/A",
                f"({metrics_b.apex_point[0]}, {metrics_b.apex_point[1]})" if metrics_b.apex_point else "N/A",
                f"({metrics_b.release_point[0]}, {metrics_b.release_point[1]})" if metrics_b.release_point else "N/A",
                f"{metrics_b.flight_duration_sec:.2f} s",
                f"{metrics_b.detected_frames_count} / {metrics_b.total_frames_analyzed}",
                f"{metrics_b.tracking_success_rate:.1f}%",
                f"{metrics_b.avg_speed_px_per_sec:.1f} px/s" if metrics_b.avg_speed_px_per_sec is not None else "N/A",
            ]
        }
        return pd.DataFrame(data)

    @staticmethod
    def generate_comparison_plot(
        dets_a: List[BallDetection],
        metrics_a: ShotMetrics,
        name_a: str,
        dets_b: List[BallDetection],
        metrics_b: ShotMetrics,
        name_b: str,
        frame_width: int = 1280,
        frame_height: int = 720,
    ) -> plt.Figure:
        """
        Creates an overlay plot comparing both trajectories on the same coordinate plane.
        """
        fig, ax = plt.subplots(figsize=(9, 5.5), dpi=100)
        fig.patch.set_facecolor('#0E1117')
        ax.set_facecolor('#1E212B')

        # Shot A: Cyan / Teal
        valid_a = [d for d in dets_a if d.detected and d.center is not None]
        if valid_a:
            xs_a = [d.center[0] for d in valid_a]
            ys_a = [d.center[1] for d in valid_a]
            ax.plot(xs_a, ys_a, color='#00E5FF', linewidth=2.5, label=f'{name_a} Trajectory', alpha=0.9)
            ax.scatter(xs_a[0], ys_a[0], color='#00E5FF', s=100, marker='o', edgecolors='white', label=f'{name_a} Release')
            if metrics_a.apex_point:
                ax.scatter(metrics_a.apex_point[0], metrics_a.apex_point[1], color='#00E5FF', s=120,
                           marker='^', edgecolors='white', label=f'{name_a} Apex')

        # Shot B: Orange / Red
        valid_b = [d for d in dets_b if d.detected and d.center is not None]
        if valid_b:
            xs_b = [d.center[0] for d in valid_b]
            ys_b = [d.center[1] for d in valid_b]
            ax.plot(xs_b, ys_b, color='#FF6D00', linewidth=2.5, linestyle='--', label=f'{name_b} Trajectory', alpha=0.9)
            ax.scatter(xs_b[0], ys_b[0], color='#FF6D00', s=100, marker='s', edgecolors='white', label=f'{name_b} Release')
            if metrics_b.apex_point:
                ax.scatter(metrics_b.apex_point[0], metrics_b.apex_point[1], color='#FF6D00', s=120,
                           marker='v', edgecolors='white', label=f'{name_b} Apex')

        # Hoop markers (Shot A hoop and/or Shot B hoop)
        if metrics_a.hoop_center:
            hx, hy = metrics_a.hoop_center
            hoop_circle = plt.Circle((hx, hy), 25, color='#E0E0E0', fill=False,
                                     linewidth=2.0, linestyle=':', label='Target Hoop')
            ax.add_patch(hoop_circle)
            ax.scatter(hx, hy, color='#E0E0E0', s=60, marker='+')

        # Coordinate axes: Invert Y-axis for image plane
        ax.set_xlim(0, frame_width)
        ax.set_ylim(frame_height, 0)

        ax.set_title("Shot Trajectory Comparison (2D Image Plane)", color='#FFFFFF', fontsize=13, fontweight='bold', pad=12)
        ax.set_xlabel("Horizontal Position X (pixels)", color='#B0BEC5', fontsize=10)
        ax.set_ylabel("Vertical Position Y (pixels)", color='#B0BEC5', fontsize=10)
        ax.tick_params(colors='#B0BEC5', labelsize=9)
        ax.grid(True, color='#2C3240', linestyle=':', alpha=0.8)

        legend = ax.legend(facecolor='#14171F', edgecolor='#374151', fontsize=8, loc='upper right')
        for text in legend.get_texts():
            text.set_color('#E0E0E0')

        fig.tight_layout()
        return fig

    @staticmethod
    def generate_analytical_insights(
        metrics_a: ShotMetrics,
        name_a: str,
        metrics_b: ShotMetrics,
        name_b: str
    ) -> List[str]:
        """
        Generates objective comparative observations based purely on image-plane kinematics.
        Avoids making false basket-entry claims.
        """
        insights = []

        # 1. Launch Angle Comparison
        if metrics_a.launch_angle_deg is not None and metrics_b.launch_angle_deg is not None:
            angle_diff = metrics_a.launch_angle_deg - metrics_b.launch_angle_deg
            if abs(angle_diff) < 2.0:
                insights.append(
                    f"**Launch Elevation:** Both shots exhibited very similar release angles "
                    f"({metrics_a.launch_angle_deg:.1f}° vs {metrics_b.launch_angle_deg:.1f}°, diff: {abs(angle_diff):.1f}°)."
                )
            elif angle_diff > 0:
                insights.append(
                    f"**Launch Elevation:** **{name_a}** had a steeper launch angle ({metrics_a.launch_angle_deg:.1f}°) "
                    f"compared to **{name_b}** ({metrics_b.launch_angle_deg:.1f}°), yielding a steeper entry arc."
                )
            else:
                insights.append(
                    f"**Launch Elevation:** **{name_b}** had a steeper launch angle ({metrics_b.launch_angle_deg:.1f}°) "
                    f"compared to **{name_a}** ({metrics_a.launch_angle_deg:.1f}°)."
                )

        # 2. Apex / Height Comparison
        if metrics_a.apex_point and metrics_b.apex_point:
            # Note: lower Y in image plane means higher physically
            y_a = metrics_a.apex_point[1]
            y_b = metrics_b.apex_point[1]
            if y_a < y_b:
                diff_px = y_b - y_a
                insights.append(
                    f"**Arc Height:** **{name_a}** achieved a higher trajectory peak in the image frame "
                    f"(apex Y: {y_a} px vs {y_b} px, {diff_px:.0f} px higher)."
                )
            else:
                diff_px = y_a - y_b
                insights.append(
                    f"**Arc Height:** **{name_b}** reached a higher trajectory peak in the image frame "
                    f"(apex Y: {y_b} px vs {y_a} px, {diff_px:.0f} px higher)."
                )

        # 3. Hoop Proximity Comparison
        if metrics_a.closest_distance_px is not None and metrics_b.closest_distance_px is not None:
            dist_diff = metrics_a.closest_distance_px - metrics_b.closest_distance_px
            if abs(dist_diff) < 5.0:
                insights.append(
                    f"**Hoop Proximity:** Both trajectories passed at nearly identical image-plane distances to the hoop center "
                    f"({metrics_a.closest_distance_px:.1f} px vs {metrics_b.closest_distance_px:.1f} px)."
                )
            elif dist_diff < 0:
                insights.append(
                    f"**Hoop Proximity:** **{name_a}** passed closer to the selected hoop center "
                    f"({metrics_a.closest_distance_px:.1f} px) than **{name_b}** ({metrics_b.closest_distance_px:.1f} px)."
                )
            else:
                insights.append(
                    f"**Hoop Proximity:** **{name_b}** passed closer to the selected hoop center "
                    f"({metrics_b.closest_distance_px:.1f} px) than **{name_a}** ({metrics_a.closest_distance_px:.1f} px)."
                )

        # 4. Flight Duration
        if metrics_a.flight_duration_sec > 0 and metrics_b.flight_duration_sec > 0:
            dur_diff = metrics_a.flight_duration_sec - metrics_b.flight_duration_sec
            insights.append(
                f"**Flight Hang Time:** {name_a} was tracked for {metrics_a.flight_duration_sec:.2f} s; "
                f"{name_b} was tracked for {metrics_b.flight_duration_sec:.2f} s."
            )

        insights.append(
            "> ⚠️ *Computer Vision Note: Distance metrics are computed strictly in 2D image-plane pixels. "
            "Without multi-camera 3D stereo calibration, proximity does not confirm true 3D spatial hoop penetration or basket scoring.*"
        )

        return insights
