"""
video_processing.py
CourtVision - Video Processing & Visual Annotation Pipeline

Handles:
1. Video metadata extraction and frame sampling
2. Video frame-by-frame ball tracking execution with progress reporting
3. Visual HUD annotations (Trajectory polyline, hoop ring, ball tracker, telemetry)
4. Video writing and CSV dataset export
"""

import cv2
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Callable, Dict, Any
from ball_tracker import BallTracker, BallDetection
from trajectory_analyzer import ShotMetrics


class VideoProcessor:
    """
    Orchestrates video reading, ball detection execution, telemetry rendering,
    and annotated video generation.
    """

    def __init__(self, video_path: str):
        self.video_path = video_path
        self._inspect_video()

    def _inspect_video(self):
        """Reads video metadata to verify integrity and retrieve properties."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file at path: {self.video_path}")

        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(cap.get(cv2.CAP_PROP_FPS))
        if self.fps <= 0 or np.isnan(self.fps):
            self.fps = 30.0  # Fallback default

        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.total_frames <= 0:
            self.total_frames = 1

        self.duration_sec = self.total_frames / self.fps
        cap.release()

    def get_frame(self, frame_idx: int = 0) -> Optional[np.ndarray]:
        """
        Retrieves a specific video frame as an RGB image for Streamlit display.
        Returns None if frame cannot be read.
        """
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return None

        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(frame_idx, self.total_frames - 1)))
        ret, frame_bgr = cap.read()
        cap.release()

        if not ret or frame_bgr is None:
            return None

        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def draw_annotations(
        self,
        frame_bgr: np.ndarray,
        current_det: BallDetection,
        tracked_history: List[BallDetection],
        hoop_center: Optional[Tuple[int, int]] = None,
        hoop_radius: float = 25.0,
        launch_angle: Optional[float] = None,
        closest_dist: Optional[float] = None,
    ) -> np.ndarray:
        """
        Overlays professional sports-broadcast visual annotations on a video frame:
        - Trajectory polyline trail with glow effect
        - Basketball tracking target circle and crosshairs
        - Hoop target ring with pulse
        - Telemetry HUD banner with frame stats
        """
        annotated = frame_bgr.copy()
        h, w = annotated.shape[:2]

        # 1. Draw Trajectory Trail
        valid_points = [
            d.center for d in tracked_history
            if d.detected and d.center is not None and d.frame_idx <= current_det.frame_idx
        ]

        if len(valid_points) >= 2:
            pts_array = np.array(valid_points, dtype=np.int32).reshape((-1, 1, 2))
            # Outer cyan glow
            cv2.polylines(annotated, [pts_array], isClosed=False, color=(255, 230, 0), thickness=4, lineType=cv2.LINE_AA)
            # Inner white/cyan core
            cv2.polylines(annotated, [pts_array], isClosed=False, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

            # Draw small fading dots along recent points
            for pt in valid_points[-12:]:
                cv2.circle(annotated, pt, 3, (0, 200, 255), -1, lineType=cv2.LINE_AA)

        # 2. Draw Hoop Target
        if hoop_center is not None:
            hx, hy = hoop_center
            hr = int(hoop_radius)
            # Rim outer ring
            cv2.circle(annotated, (hx, hy), hr, (0, 0, 255), 2, lineType=cv2.LINE_AA)
            # Center crosshair
            cv2.line(annotated, (hx - 8, hy), (hx + 8, hy), (0, 0, 255), 1, lineType=cv2.LINE_AA)
            cv2.line(annotated, (hx, hy - 8), (hx, hy + 8), (0, 0, 255), 1, lineType=cv2.LINE_AA)
            # Hoop Label
            cv2.putText(annotated, "HOOP", (hx - 20, hy - hr - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

        # 3. Draw Current Ball Detection
        if current_det.detected and current_det.center is not None:
            bx, by = current_det.center
            br = int(current_det.radius) if current_det.radius else 15

            # Green tracking circle
            cv2.circle(annotated, (bx, by), max(8, br + 4), (0, 255, 0), 2, lineType=cv2.LINE_AA)
            cv2.circle(annotated, (bx, by), 3, (0, 255, 0), -1, lineType=cv2.LINE_AA)

            # Connect line to hoop if in proximity
            if hoop_center is not None:
                dist_now = np.hypot(bx - hoop_center[0], by - hoop_center[1])
                if dist_now < 200:
                    cv2.line(annotated, (bx, by), hoop_center, (0, 165, 255), 1, cv2.LINE_AA)
                    cv2.putText(annotated, f"{int(dist_now)}px", ((bx + hoop_center[0]) // 2, (by + hoop_center[1]) // 2 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1, cv2.LINE_AA)

        # 4. Draw HUD Telemetry Header
        hud_h = 42
        hud_overlay = annotated.copy()
        cv2.rectangle(hud_overlay, (0, 0), (w, hud_h), (20, 20, 25), -1)
        cv2.addWeighted(hud_overlay, 0.75, annotated, 0.25, 0, annotated)

        # Telemetry text
        cv2.putText(annotated, "COURTVISION ANALYTICS", (15, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 255), 2, cv2.LINE_AA)

        hud_info = f"Frame: {current_det.frame_idx:03d} | Time: {current_det.timestamp:.2f}s"
        if launch_angle is not None:
            hud_info += f" | Launch: {launch_angle:.1f} deg"
        if closest_dist is not None:
            hud_info += f" | Min Dist: {closest_dist:.1f}px"

        cv2.putText(annotated, hud_info, (240, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

        return annotated

    def process_and_annotate(
        self,
        tracker: BallTracker,
        output_video_path: Optional[str] = None,
        hoop_center: Optional[Tuple[int, int]] = None,
        hoop_radius: float = 25.0,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Tuple[List[BallDetection], Optional[str]]:
        """
        Runs the full detection and annotation pipeline through the video.

        Args:
            tracker: BallTracker instance to use.
            output_video_path: Filepath where annotated video will be saved.
            hoop_center: Coordinates of selected hoop.
            hoop_radius: Radius of hoop target.
            progress_callback: Optional function(current_frame, total_frames) for progress UI.

        Returns:
            Tuple of (list of BallDetections, path to saved output video or None)
        """
        tracker.reset()
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {self.video_path}")

        # Setup Video Writer if output path requested
        writer = None
        if output_video_path:
            # On Windows, 'mp4v' or 'avc1' are standard FourCC codecs
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_video_path, fourcc, self.fps, (self.width, self.height))

        detections: List[BallDetection] = []
        frame_idx = 0

        raw_frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            timestamp = frame_idx / self.fps
            det, _ = tracker.process_frame(frame, frame_idx, timestamp)
            detections.append(det)

            if writer is not None:
                raw_frames.append(frame)

            frame_idx += 1
            if progress_callback and frame_idx % 5 == 0:
                progress_callback(frame_idx, self.total_frames)

        cap.release()

        # Smooth and filter trajectory to eliminate single-frame jumps
        detections = self.filter_and_smooth_detections(detections)

        # Compute preliminary launch angle and min distance for video overlay
        valid_coords = [d.center for d in detections if d.detected and d.center is not None]
        launch_angle = None
        closest_dist = None

        if len(valid_coords) >= 4:
            xs = np.array([c[0] for c in valid_coords])
            ys = np.array([c[1] for c in valid_coords])
            dx = abs(xs[min(6, len(xs)-1)] - xs[0])
            dy_up = -(ys[min(6, len(ys)-1)] - ys[0])
            launch_angle = round(float(np.degrees(np.arctan2(dy_up, dx))), 1)

        if hoop_center is not None and valid_coords:
            hx, hy = hoop_center
            dists = [np.hypot(c[0] - hx, c[1] - hy) for c in valid_coords]
            closest_dist = round(float(min(dists)), 1)

        # If writing annotated video, render overlays frame-by-frame
        if writer is not None:
            for idx, frame in enumerate(raw_frames):
                det = detections[idx]
                annotated = self.draw_annotations(
                    frame_bgr=frame,
                    current_det=det,
                    tracked_history=detections[:idx + 1],
                    hoop_center=hoop_center,
                    hoop_radius=hoop_radius,
                    launch_angle=launch_angle,
                    closest_dist=closest_dist,
                )
                writer.write(annotated)
            writer.release()

        if progress_callback:
            progress_callback(self.total_frames, self.total_frames)

        return detections, output_video_path

    @staticmethod
    def filter_and_smooth_detections(detections: List[BallDetection], max_jump: float = 120.0) -> List[BallDetection]:
        """
        Removes isolated single-frame outlier jumps and smooths the trajectory
        to ensure continuous physical projectile motion.
        """
        cleaned = list(detections)
        valid_indices = [i for i, d in enumerate(cleaned) if d.detected and d.center is not None]

        if len(valid_indices) < 3:
            return cleaned

        for k in range(1, len(valid_indices) - 1):
            prev_i = valid_indices[k - 1]
            curr_i = valid_indices[k]
            next_i = valid_indices[k + 1]

            curr_pt = cleaned[curr_i].center
            prev_pt = cleaned[prev_i].center
            next_pt = cleaned[next_i].center

            d_prev = np.hypot(curr_pt[0] - prev_pt[0], curr_pt[1] - prev_pt[1])
            d_next = np.hypot(curr_pt[0] - next_pt[0], curr_pt[1] - next_pt[1])
            d_span = np.hypot(next_pt[0] - prev_pt[0], next_pt[1] - prev_pt[1])

            # If current point spiked drastically away while previous and next are close
            if d_prev > max_jump and d_next > max_jump and d_span < max_jump * 1.6:
                # Interpolate between previous and next
                t_ratio = (curr_i - prev_i) / max(1, next_i - prev_i)
                ix = int(prev_pt[0] + t_ratio * (next_pt[0] - prev_pt[0]))
                iy = int(prev_pt[1] + t_ratio * (next_pt[1] - prev_pt[1]))
                cleaned[curr_i].center = (ix, iy)
                cleaned[curr_i].is_interpolated = True

        return cleaned

    @staticmethod
    def export_csv(
        detections: List[BallDetection],
        hoop_center: Optional[Tuple[int, int]] = None,
        output_csv_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Exports ball coordinates, timestamps, and hoop distance metrics to a DataFrame / CSV file.
        """
        records = []
        for d in detections:
            cx = d.center[0] if d.center else None
            cy = d.center[1] if d.center else None
            dist_hoop = None
            if hoop_center and cx is not None and cy is not None:
                dist_hoop = round(float(np.hypot(cx - hoop_center[0], cy - hoop_center[1])), 2)

            records.append({
                "frame_index": d.frame_idx,
                "timestamp_sec": round(d.timestamp, 4),
                "detected": 1 if d.detected else 0,
                "center_x_px": cx,
                "center_y_px": cy,
                "radius_px": round(d.radius, 2) if d.radius else None,
                "area_px2": round(d.area, 2) if d.area else None,
                "circularity": round(d.circularity, 4) if d.circularity else None,
                "distance_to_hoop_px": dist_hoop,
            })

        df = pd.DataFrame(records)
        if output_csv_path:
            df.to_csv(output_csv_path, index=False)
        return df
