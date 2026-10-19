"""
trajectory_analyzer.py
CourtVision - Trajectory Analysis and Kinematics Module

Calculates:
1. Release point identification from motion initiation
2. Launch / release angle estimation using multi-frame vector regression
3. Trajectory apex (highest image-plane elevation)
4. Image-plane distance and relative displacement to the hoop
5. Time-series and 2D spatial trajectory plots using Matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from ball_tracker import BallDetection


@dataclass
class ShotMetrics:
    """Quantitative image-plane metrics computed for a single basketball shot."""
    total_frames_analyzed: int
    detected_frames_count: int
    tracking_success_rate: float  # Percentage of frames with successful detection
    flight_duration_sec: float  # Duration from release to last detection
    release_frame_idx: Optional[int] = None
    release_point: Optional[Tuple[int, int]] = None  # (x, y) pixels
    release_time_sec: Optional[float] = None
    launch_angle_deg: Optional[float] = None  # Elevation angle relative to horizontal
    apex_point: Optional[Tuple[int, int]] = None  # (x, y) pixels (min y)
    apex_frame_idx: Optional[int] = None
    apex_time_sec: Optional[float] = None
    hoop_center: Optional[Tuple[int, int]] = None
    closest_distance_px: Optional[float] = None  # Min Euclidean distance in pixels
    closest_point: Optional[Tuple[int, int]] = None
    closest_frame_idx: Optional[int] = None
    closest_time_sec: Optional[float] = None
    relative_pos_at_closest: Optional[Tuple[int, int]] = None  # (dx, dy) px relative to hoop
    avg_speed_px_per_sec: Optional[float] = None


class TrajectoryAnalyzer:
    """
    Performs kinematic analysis on detected basketball coordinates.
    All distance metrics are strictly computed in 2D image-plane pixels.
    """

    def __init__(self, hoop_center: Optional[Tuple[int, int]] = None, hoop_radius: float = 25.0):
        """
        Args:
            hoop_center: (x, y) center pixel coordinates of the hoop region.
            hoop_radius: Estimated radius of the hoop in pixels.
        """
        self.hoop_center = hoop_center
        self.hoop_radius = hoop_radius

    def set_hoop(self, center: Tuple[int, int], radius: float = 25.0):
        """Update the designated hoop location."""
        self.hoop_center = center
        self.hoop_radius = radius

    def analyze(self, detections: List[BallDetection]) -> ShotMetrics:
        """
        Extract trajectory metrics from the sequence of ball detections.

        Args:
            detections: List of BallDetection objects across video frames.

        Returns:
            ShotMetrics object with calculated kinematic values.
        """
        valid_dets = [d for d in detections if d.detected and d.center is not None]
        total_frames = len(detections)
        detected_count = len(valid_dets)

        if detected_count < 3:
            return ShotMetrics(
                total_frames_analyzed=total_frames,
                detected_frames_count=detected_count,
                tracking_success_rate=(detected_count / total_frames * 100.0) if total_frames > 0 else 0.0,
                flight_duration_sec=0.0,
            )

        # Extract coordinate and time arrays
        times = np.array([d.timestamp for d in valid_dets], dtype=float)
        xs = np.array([d.center[0] for d in valid_dets], dtype=float)
        ys = np.array([d.center[1] for d in valid_dets], dtype=float)
        frame_indices = [d.frame_idx for d in valid_dets]

        # 1. Release Point: First detected moving frame
        # In a shot sequence, the release is the initial point of the tracked trajectory
        release_idx = 0
        release_pt = (int(xs[0]), int(ys[0]))
        release_frame = frame_indices[0]
        release_time = float(times[0])

        # Flight duration (from first to last tracked frame)
        flight_duration = float(times[-1] - times[0])

        # 2. Launch Angle Estimation
        # Take the first N detections (up to 8 points or 30% of trajectory) to estimate initial launch vector
        sample_count = min(max(4, int(len(valid_dets) * 0.3)), min(10, len(valid_dets)))
        launch_angle = self._estimate_launch_angle(xs[:sample_count], ys[:sample_count])

        # 3. Apex Detection (Highest point in trajectory)
        # Note: In computer vision image coordinates, Y increases downward.
        # Therefore, highest elevation corresponds to minimum Y value.
        apex_sub_idx = int(np.argmin(ys))
        apex_pt = (int(xs[apex_sub_idx]), int(ys[apex_sub_idx]))
        apex_frame = frame_indices[apex_sub_idx]
        apex_time = float(times[apex_sub_idx])

        # 4. Image-Plane Distance to Hoop
        closest_dist = None
        closest_pt = None
        closest_frame = None
        closest_time = None
        rel_pos = None

        if self.hoop_center is not None:
            hx, hy = self.hoop_center
            distances = np.hypot(xs - hx, ys - hy)
            min_dist_idx = int(np.argmin(distances))
            closest_dist = float(distances[min_dist_idx])
            closest_pt = (int(xs[min_dist_idx]), int(ys[min_dist_idx]))
            closest_frame = frame_indices[min_dist_idx]
            closest_time = float(times[min_dist_idx])
            rel_pos = (int(xs[min_dist_idx] - hx), int(ys[min_dist_idx] - hy))

        # 5. Average speed in image plane (px/sec)
        dxs = np.diff(xs)
        dys = np.diff(ys)
        dts = np.diff(times)
        valid_intervals = dts > 0.001
        if np.any(valid_intervals):
            segment_dists = np.hypot(dxs[valid_intervals], dys[valid_intervals])
            total_dist = float(np.sum(segment_dists))
            avg_speed = total_dist / flight_duration if flight_duration > 0.0 else 0.0
        else:
            avg_speed = 0.0

        return ShotMetrics(
            total_frames_analyzed=total_frames,
            detected_frames_count=detected_count,
            tracking_success_rate=(detected_count / total_frames * 100.0) if total_frames > 0 else 0.0,
            flight_duration_sec=flight_duration,
            release_frame_idx=release_frame,
            release_point=release_pt,
            release_time_sec=release_time,
            launch_angle_deg=launch_angle,
            apex_point=apex_pt,
            apex_frame_idx=apex_frame,
            apex_time_sec=apex_time,
            hoop_center=self.hoop_center,
            closest_distance_px=closest_dist,
            closest_point=closest_pt,
            closest_frame_idx=closest_frame,
            closest_time_sec=closest_time,
            relative_pos_at_closest=rel_pos,
            avg_speed_px_per_sec=avg_speed,
        )

    def _estimate_launch_angle(self, xs: np.ndarray, ys: np.ndarray) -> float:
        """
        Estimates the initial elevation launch angle in degrees.
        Uses linear regression / directional displacement on the initial segment.
        Image Y grows downwards, so upward physical movement means -dY > 0.
        Angle is measured with respect to the horizontal plane (0 deg = horizontal, 90 deg = vertical).
        """
        if len(xs) < 2:
            return 0.0

        # Fit linear polynomial or use displacement vector
        # dx is horizontal motion, -dy is upward vertical motion
        dx = xs[-1] - xs[0]
        dy_upward = -(ys[-1] - ys[0])

        # If shooter is moving left-to-right, dx > 0; if right-to-left, dx < 0.
        # We consider the elevation angle with respect to forward horizontal movement:
        abs_dx = abs(dx)

        # Angle in degrees
        angle_rad = np.arctan2(dy_upward, abs_dx)
        angle_deg = float(np.degrees(angle_rad))

        # Clamp to realistic range [0, 90] degrees
        return round(float(np.clip(angle_deg, 0.0, 90.0)), 1)

    def generate_trajectory_plot(
        self,
        detections: List[BallDetection],
        frame_width: int,
        frame_height: int,
        metrics: Optional[ShotMetrics] = None,
        title: str = "Ball Trajectory (2D Image Plane)"
    ) -> plt.Figure:
        """
        Creates a 2D spatial trajectory plot matching the video frame coordinates.
        Inverts Y-axis to replicate camera perspective.
        """
        fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
        fig.patch.set_facecolor('#0E1117')
        ax.set_facecolor('#1E212B')

        valid_dets = [d for d in detections if d.detected and d.center is not None]
        if valid_dets:
            xs = [d.center[0] for d in valid_dets]
            ys = [d.center[1] for d in valid_dets]

            # Trajectory polyline
            ax.plot(xs, ys, color='#00E5FF', linewidth=2.5, linestyle='-', label='Trajectory Path', alpha=0.9)
            # Scatter points
            ax.scatter(xs, ys, color='#FF9100', s=25, alpha=0.7, edgecolors='none', label='Detections')

            # Highlight Release Point
            ax.scatter(xs[0], ys[0], color='#00E676', s=120, marker='o', edgecolors='white',
                       linewidth=1.5, zorder=5, label='Release Point')

            # Highlight Apex
            if metrics and metrics.apex_point:
                ax.scatter(metrics.apex_point[0], metrics.apex_point[1], color='#FFD600', s=140,
                           marker='^', edgecolors='white', linewidth=1.5, zorder=5, label='Apex (Peak)')

            # Highlight Closest Point to Hoop
            if metrics and metrics.closest_point and metrics.hoop_center:
                ax.scatter(metrics.closest_point[0], metrics.closest_point[1], color='#FF1744', s=130,
                           marker='X', edgecolors='white', linewidth=1.5, zorder=5, label='Closest to Hoop')
                # Draw connecting vector line to hoop
                ax.plot([metrics.closest_point[0], metrics.hoop_center[0]],
                        [metrics.closest_point[1], metrics.hoop_center[1]],
                        color='#FF1744', linestyle='--', linewidth=1.5, alpha=0.8,
                        label=f'Min Dist ({metrics.closest_distance_px:.1f} px)')

        # Draw Hoop
        if self.hoop_center is not None:
            hx, hy = self.hoop_center
            hoop_circle = plt.Circle((hx, hy), self.hoop_radius, color='#FF5252', fill=False,
                                     linewidth=2.5, linestyle='-', label='Hoop Target')
            ax.add_patch(hoop_circle)
            ax.scatter(hx, hy, color='#FF5252', s=50, marker='+', zorder=6)

        # Coordinate setup: Invert Y-axis so (0,0) is top-left, matching image coordinates
        ax.set_xlim(0, frame_width)
        ax.set_ylim(frame_height, 0)

        ax.set_title(title, color='#FFFFFF', fontsize=13, fontweight='bold', pad=12)
        ax.set_xlabel("Horizontal Position X (pixels)", color='#B0BEC5', fontsize=10)
        ax.set_ylabel("Vertical Position Y (pixels - Top to Bottom)", color='#B0BEC5', fontsize=10)
        ax.tick_params(colors='#B0BEC5', labelsize=9)
        ax.grid(True, color='#2C3240', linestyle=':', alpha=0.8)

        # Legend with dark theme
        legend = ax.legend(facecolor='#14171F', edgecolor='#374151', fontsize=8, loc='upper right')
        for text in legend.get_texts():
            text.set_color('#E0E0E0')

        fig.tight_layout()
        return fig

    def generate_kinematics_plot(
        self,
        detections: List[BallDetection],
        metrics: Optional[ShotMetrics] = None,
        title: str = "Ball Kinematics over Time"
    ) -> plt.Figure:
        """
        Creates time-series charts of horizontal position X(t), vertical position Y(t),
        and distance to the hoop d(t).
        """
        valid_dets = [d for d in detections if d.detected and d.center is not None]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, dpi=100)
        fig.patch.set_facecolor('#0E1117')

        if not valid_dets:
            ax1.set_facecolor('#1E212B')
            ax2.set_facecolor('#1E212B')
            ax1.text(0.5, 0.5, "No tracking data available", color='#9E9E9E', ha='center', va='center')
            return fig

        times = [d.timestamp for d in valid_dets]
        xs = [d.center[0] for d in valid_dets]
        ys = [d.center[1] for d in valid_dets]

        # Top chart: X and Y vs Time
        ax1.set_facecolor('#1E212B')
        ax1.plot(times, xs, color='#00E5FF', linewidth=2.0, label='X Position (px)')
        # Invert Y values display visually or plot directly
        ax1.plot(times, ys, color='#FF9100', linewidth=2.0, label='Y Position (px, lower is higher)')
        ax1.set_ylabel("Pixel Coordinate", color='#B0BEC5', fontsize=10)
        ax1.set_title("Position Components X(t) and Y(t)", color='#FFFFFF', fontsize=11, fontweight='bold')
        ax1.tick_params(colors='#B0BEC5', labelsize=9)
        ax1.grid(True, color='#2C3240', linestyle=':', alpha=0.8)
        leg1 = ax1.legend(facecolor='#14171F', edgecolor='#374151', fontsize=8, loc='best')
        for text in leg1.get_texts():
            text.set_color('#E0E0E0')

        # Bottom chart: Distance to Hoop vs Time
        ax2.set_facecolor('#1E212B')
        if self.hoop_center is not None:
            hx, hy = self.hoop_center
            dists = [np.hypot(x - hx, y - hy) for x, y in zip(xs, ys)]
            ax2.plot(times, dists, color='#FF5252', linewidth=2.0, label='Distance to Hoop Center (px)')

            if metrics and metrics.closest_distance_px is not None and metrics.closest_time_sec is not None:
                ax2.scatter(metrics.closest_time_sec, metrics.closest_distance_px, color='#00E676',
                            s=90, zorder=5, label=f'Closest Approach ({metrics.closest_distance_px:.1f} px)')
                ax2.axhline(metrics.closest_distance_px, color='#00E676', linestyle='--', alpha=0.5)

            ax2.set_ylabel("Distance (pixels)", color='#B0BEC5', fontsize=10)
            ax2.set_title("Proximity to Selected Hoop over Time", color='#FFFFFF', fontsize=11, fontweight='bold')
            ax2.grid(True, color='#2C3240', linestyle=':', alpha=0.8)
            leg2 = ax2.legend(facecolor='#14171F', edgecolor='#374151', fontsize=8, loc='best')
            for text in leg2.get_texts():
                text.set_color('#E0E0E0')
        else:
            ax2.text(0.5, 0.5, "Hoop not selected", color='#9E9E9E', ha='center', va='center')

        ax2.set_xlabel("Time (seconds)", color='#B0BEC5', fontsize=10)
        ax2.tick_params(colors='#B0BEC5', labelsize=9)

        fig.tight_layout()
        return fig
