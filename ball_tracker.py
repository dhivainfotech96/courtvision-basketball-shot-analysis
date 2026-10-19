"""
ball_tracker.py
CourtVision - Basketball Detection & Tracking Module

Engineered for both practice court footage and real-time NBA broadcast clips:
1. Dual HSV Presets: Real NBA Broadcast (handles compressed lighting & dim crowds) and High-Contrast
2. Circularity and Aspect-ratio filtering resilient to fast motion blur
3. Seeded Initial Ball Tracking (locks onto shooter release to ignore defender jerseys)
4. Motion frame differencing to reject static crowd spectators and advertisements
5. Kinematic velocity gating with gravitational coasting to prevent crowd teleportation
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any


@dataclass
class BallDetection:
    """Represents a single frame's ball detection result."""
    frame_idx: int
    timestamp: float
    detected: bool
    center: Optional[Tuple[int, int]] = None
    radius: Optional[float] = None
    area: Optional[float] = None
    circularity: Optional[float] = None
    bbox: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
    is_interpolated: bool = False


class BallTracker:
    """
    Detects and tracks a basketball across video frames.
    Supports real-world NBA broadcast videos with dynamic camera motion,
    crowd spectators, and motion-blurred basketballs.
    """

    def __init__(
        self,
        mode: str = "practice_court",
        hsv_lower: Optional[Tuple[int, int, int]] = None,
        hsv_upper: Optional[Tuple[int, int, int]] = None,
        min_area: float = 40.0,
        max_area: float = 12000.0,
        min_circularity: float = 0.35,
        max_aspect_ratio_diff: float = 0.65,
        max_jump_distance: float = 180.0,
        initial_ball_pos: Optional[Tuple[int, int]] = None,
        initial_frame_idx: int = 0,
        use_motion_filter: bool = False,
    ):
        """
        Args:
            mode: 'nba_broadcast' (optimized for real game footage) or 'practice_court'.
            hsv_lower: Optional custom lower HSV bound.
            hsv_upper: Optional custom upper HSV bound.
            min_area: Minimum contour area.
            max_area: Maximum contour area.
            min_circularity: Minimum circularity.
            max_jump_distance: Maximum distance a ball can travel between frames.
            initial_ball_pos: Optional (x, y) starting position to lock onto shooter's release.
            initial_frame_idx: Frame index where initial position is seeded.
            use_motion_filter: Whether to apply temporal differencing to suppress static crowd.
        """
        self.mode = mode

        # Set default HSV ranges based on preset mode
        if hsv_lower is not None and hsv_upper is not None:
            self.hsv_lower = np.array(hsv_lower, dtype=np.uint8)
            self.hsv_upper = np.array(hsv_upper, dtype=np.uint8)
        elif mode == "nba_broadcast":
            # Real NBA broadcast: captures Spalding/Wilson leather in stadium illumination
            self.hsv_lower = np.array([5, 40, 32], dtype=np.uint8)
            self.hsv_upper = np.array([25, 255, 245], dtype=np.uint8)
        else:
            # Practice court / clean camera
            self.hsv_lower = np.array([5, 95, 80], dtype=np.uint8)
            self.hsv_upper = np.array([25, 255, 255], dtype=np.uint8)

        self.min_area = min_area
        self.max_area = max_area
        self.min_circularity = min_circularity
        self.max_aspect_ratio_diff = max_aspect_ratio_diff
        self.max_jump_distance = max_jump_distance
        self.initial_ball_pos = initial_ball_pos
        self.initial_frame_idx = initial_frame_idx
        self.use_motion_filter = use_motion_filter

        # Tracking state
        self.last_position: Optional[Tuple[int, int]] = None
        self.last_velocity: Tuple[float, float] = (0.0, 0.0)
        self.lost_frame_count: int = 0
        self.history: List[BallDetection] = []
        self.prev_gray_frame: Optional[np.ndarray] = None

    def reset(self):
        """Reset internal tracker state."""
        self.last_position = self.initial_ball_pos
        self.last_velocity = (0.0, 0.0)
        self.lost_frame_count = 0
        self.history.clear()
        self.prev_gray_frame = None

    def create_color_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Isolates orange basketball pixels using HSV thresholding and morphology."""
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        if self.hsv_lower[0] <= self.hsv_upper[0]:
            mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        else:
            lower1 = np.array([0, self.hsv_lower[1], self.hsv_lower[2]], dtype=np.uint8)
            upper1 = np.array([self.hsv_upper[0], self.hsv_upper[1], self.hsv_upper[2]], dtype=np.uint8)
            lower2 = np.array([self.hsv_lower[0], self.hsv_lower[1], self.hsv_lower[2]], dtype=np.uint8)
            upper2 = np.array([179, self.hsv_upper[1], self.hsv_upper[2]], dtype=np.uint8)
            mask = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1), cv2.inRange(hsv, lower2, upper2))

        # Morphological filtering: open removes noise, close bridges seams & reflections
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)

        return mask

    def _compute_circularity(self, contour: np.ndarray, area: float) -> float:
        """Circularity quotient = 4 * pi * Area / Perimeter^2."""
        perimeter = cv2.arcLength(contour, closed=True)
        if perimeter <= 0:
            return 0.0
        return float((4.0 * np.pi * area) / (perimeter * perimeter))

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        frame_idx: int,
        timestamp: float,
        return_mask: bool = False
    ) -> Tuple[BallDetection, Optional[np.ndarray]]:
        """
        Process a single video frame with color segmentation, motion weighting,
        and kinematic trajectory continuity.
        """
        curr_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        mask = self.create_color_mask(frame_bgr)

        # Optional motion differencing to suppress static crowd & stationary banners
        motion_mask = None
        if self.use_motion_filter and self.prev_gray_frame is not None:
            frame_diff = cv2.absdiff(curr_gray, self.prev_gray_frame)
            _, motion_mask = cv2.threshold(frame_diff, 12, 255, cv2.THRESH_BINARY)
            motion_mask = cv2.dilate(motion_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=2)

        self.prev_gray_frame = curr_gray

        # Extract contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (self.min_area <= area <= self.max_area):
                continue

            circularity = self._compute_circularity(cnt, area)
            if circularity < self.min_circularity:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / float(h) if h > 0 else 0.0
            if abs(aspect_ratio - 1.0) > self.max_aspect_ratio_diff:
                continue

            # Centroid via moments
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx = int(x + w / 2)
                cy = int(y + h / 2)

            ((_, _), radius) = cv2.minEnclosingCircle(cnt)

            # Motion score bonus: check if centroid lies in motion mask
            motion_bonus = 1.0
            if motion_mask is not None and 0 <= cy < motion_mask.shape[0] and 0 <= cx < motion_mask.shape[1]:
                if motion_mask[cy, cx] > 0:
                    motion_bonus = 1.5

            candidates.append({
                "center": (cx, cy),
                "radius": float(radius),
                "area": float(area),
                "circularity": float(circularity) * motion_bonus,
                "bbox": (x, y, w, h),
                "contour": cnt,
            })

        best_candidate = None
        is_coasting = False

        # If user seeded initial ball position on or before this frame and we haven't locked yet
        if self.last_position is None and self.initial_ball_pos is not None:
            if frame_idx >= self.initial_frame_idx:
                # Find candidate closest to user seed
                seed_x, seed_y = self.initial_ball_pos
                seeded_candidates = []
                for cand in candidates:
                    dist = np.hypot(cand["center"][0] - seed_x, cand["center"][1] - seed_y)
                    if dist <= 120.0:  # Within 120 px of user seed
                        seeded_candidates.append((dist, cand))
                if seeded_candidates:
                    seeded_candidates.sort(key=lambda item: item[0])
                    best_candidate = seeded_candidates[0][1]
                else:
                    # Initialize exactly at user seed
                    best_candidate = {
                        "center": self.initial_ball_pos,
                        "radius": 16.0,
                        "area": 800.0,
                        "circularity": 0.8,
                        "bbox": (seed_x - 16, seed_y - 16, 32, 32),
                    }

        # Active tracking logic
        if self.last_position is not None:
            # Predict position using current velocity + slight gravity downward (+0.8 px/frame^2)
            pred_x = self.last_position[0] + self.last_velocity[0]
            pred_y = self.last_position[1] + self.last_velocity[1] + (0.8 if self.last_velocity != (0.0, 0.0) else 0.0)

            # Match candidate within kinematic displacement gate
            # Dynamic gating prevents teleporting across the crowd
            allowed_jump = self.max_jump_distance * (1.0 + 0.25 * min(self.lost_frame_count, 4))
            candidates_with_dist = []

            for cand in candidates:
                cx, cy = cand["center"]
                dist = np.hypot(cx - pred_x, cy - pred_y)
                if dist <= allowed_jump:
                    # Score combines proximity and circularity
                    score = dist - (cand["circularity"] * 25.0)
                    candidates_with_dist.append((score, dist, cand))

            if candidates_with_dist:
                candidates_with_dist.sort(key=lambda item: item[0])
                best_candidate = candidates_with_dist[0][2]
            else:
                # KINEMATIC COASTING:
                # Coast along predicted trajectory for up to 3 frames if lost in crowd
                h_f, w_f = frame_bgr.shape[:2]
                if self.lost_frame_count < 4 and abs(self.last_velocity[0]) + abs(self.last_velocity[1]) > 3.0:
                    px = int(np.clip(pred_x, 0, w_f - 1))
                    py = int(np.clip(pred_y, 0, h_f - 1))
                    best_candidate = {
                        "center": (px, py),
                        "radius": 16.0,
                        "area": 800.0,
                        "circularity": 0.6,
                        "bbox": (max(0, px - 16), max(0, py - 16), 32, 32),
                    }
                    is_coasting = True
        elif best_candidate is None and candidates:
            # First initialization when no seed provided: pick highest circularity in upper 70% of frame
            valid_first = [c for c in candidates if c["center"][1] < frame_bgr.shape[0] * 0.80]
            if valid_first:
                valid_first.sort(key=lambda c: c["circularity"], reverse=True)
                best_candidate = valid_first[0]

        # Update tracking state
        if best_candidate is not None:
            cx, cy = best_candidate["center"]
            if self.last_position is not None:
                vx = cx - self.last_position[0]
                vy = cy - self.last_position[1]
                # Smooth velocity with exponential moving average
                self.last_velocity = (
                    0.65 * vx + 0.35 * self.last_velocity[0],
                    0.65 * vy + 0.35 * self.last_velocity[1],
                )
            self.last_position = (cx, cy)
            if is_coasting:
                self.lost_frame_count += 1
            else:
                self.lost_frame_count = 0

            detection = BallDetection(
                frame_idx=frame_idx,
                timestamp=timestamp,
                detected=True,
                center=(cx, cy),
                radius=best_candidate["radius"],
                area=best_candidate["area"],
                circularity=best_candidate["circularity"],
                bbox=best_candidate["bbox"],
                is_interpolated=is_coasting,
            )
        else:
            self.lost_frame_count += 1
            detection = BallDetection(
                frame_idx=frame_idx,
                timestamp=timestamp,
                detected=False,
            )

        self.history.append(detection)
        return detection, (mask if return_mask else None)

    def get_tracked_points(self) -> List[Tuple[int, int]]:
        """Returns ordered list of (x, y) coordinates for successfully detected frames."""
        return [det.center for det in self.history if det.detected and det.center is not None]
