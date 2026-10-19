"""
generate_sample_video.py
CourtVision - Synthetic Basketball Shot Video Generator

Generates high-definition, realistic basketball shooting clips with:
- Realistic arena court setting (hardwood floor, backboard, rim, net)
- Parabolic projectile motion staying cleanly within the camera frame
- Authentic textured orange basketball with seam ribs and shading
- Two distinct shot trajectories for comparison:
  1. shot_sample_1.mp4: High-arc shot (~52° launch angle)
  2. shot_sample_2.mp4: Flatter-arc shot (~40° launch angle)
"""

import cv2
import numpy as np
import os


def draw_court_background(width: int = 1280, height: int = 720, hoop_pos: tuple = (960, 260)) -> np.ndarray:
    """Draws a photorealistic indoor basketball arena background."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Arena Wall / Crowd background (dark stadium atmosphere)
    floor_y = int(height * 0.72)
    for y in range(floor_y):
        shade = int(20 + 22 * (y / floor_y))
        frame[y, :] = [shade + 8, shade + 3, shade]

    # Arena spotlight illumination
    cv2.circle(frame, (640, -50), 500, (48, 48, 60), -1)

    # Polished Hardwood Court Floor (realistic light natural wood, low saturation)
    for y in range(floor_y, height):
        ratio = (y - floor_y) / (height - floor_y)
        b = int(95 + 15 * ratio)
        g = int(135 + 20 * ratio)
        r = int(165 + 20 * ratio)
        frame[y, :] = [b, g, r]

    # Floor boundary line
    cv2.line(frame, (0, floor_y), (width, floor_y), (255, 255, 255), 3)

    # Court perspective lines
    cv2.line(frame, (200, height), (350, floor_y), (255, 255, 255), 2)
    cv2.line(frame, (900, height), (850, floor_y), (255, 255, 255), 2)

    # Free throw semi-circle outline
    cv2.ellipse(frame, (350, floor_y + 40), (140, 45), 0, 0, 180, (255, 255, 255), 2)

    # Draw Backboard & Hoop
    hx, hy = hoop_pos

    # Backboard Support Pole
    cv2.rectangle(frame, (hx + 75, hy - 140), (hx + 95, height), (60, 60, 70), -1)
    cv2.rectangle(frame, (hx + 20, hy - 25), (hx + 80, hy - 10), (90, 90, 100), -1)

    # Glass Backboard
    bb_x = hx + 10
    bb_y = hy - 110
    bb_w = 16
    bb_h = 130
    cv2.rectangle(frame, (bb_x, bb_y), (bb_x + bb_w, bb_y + bb_h), (220, 220, 230), 3)
    cv2.rectangle(frame, (bb_x + 2, hy - 50), (bb_x + 10, hy - 10), (0, 0, 220), 2)

    # Orange / Red Rim
    rim_w = 46
    rim_h = 10
    cv2.ellipse(frame, (hx - 15, hy), (rim_w // 2, rim_h // 2), 0, 0, 360, (20, 30, 230), 4)

    # Net (white mesh beneath the rim)
    net_top_y = hy + 5
    net_bottom_y = hy + 65
    net_left_x = hx - 35
    net_right_x = hx + 5
    net_mid_x = (net_left_x + net_right_x) // 2

    # Vertical net strings
    for sx in range(net_left_x, net_right_x + 1, 8):
        bot_x = int(net_mid_x + (sx - net_mid_x) * 0.5)
        cv2.line(frame, (sx, net_top_y), (bot_x, net_bottom_y), (240, 240, 240), 1)

    # Horizontal net loops
    for ny in range(net_top_y + 12, net_bottom_y, 14):
        span = int((net_right_x - net_left_x) * (1.0 - 0.4 * ((ny - net_top_y) / 60)))
        cv2.ellipse(frame, (net_mid_x, ny), (span // 2, 4), 0, 0, 180, (230, 230, 230), 1)

    return frame


def draw_realistic_basketball(frame: np.ndarray, center: tuple, radius: int = 18):
    """
    Renders an authentic orange basketball with seam grooves and spherical shading.
    Matches the HSV range: H ~ 10-18, S ~ 200-255, V ~ 200-255.
    """
    cx, cy = int(center[0]), int(center[1])
    h, w = frame.shape[:2]

    if not (0 <= cx < w and 0 <= cy < h):
        return

    # Base basketball vibrant orange (BGR: 20, 105, 245)
    for r in range(radius, 0, -1):
        ratio = r / radius
        b = int(15 + 15 * (1 - ratio))
        g = int(90 + 35 * (1 - ratio))
        r_col = int(225 + 30 * (1 - ratio))
        cv2.circle(frame, (cx - int(2 * (1 - ratio)), cy - int(2 * (1 - ratio))), r, (b, g, r_col), -1, lineType=cv2.LINE_AA)

    # Black basketball seams
    cv2.ellipse(frame, (cx, cy), (radius - 1, int(radius * 0.35)), 15, 0, 360, (25, 25, 30), 1, lineType=cv2.LINE_AA)
    cv2.ellipse(frame, (cx, cy), (int(radius * 0.35), radius - 1), 15, 0, 360, (25, 25, 30), 1, lineType=cv2.LINE_AA)
    cv2.line(frame, (cx - radius + 2, cy), (cx + radius - 2, cy), (25, 25, 30), 1, lineType=cv2.LINE_AA)


def generate_shot_video(
    output_filename: str,
    start_pos: tuple = (160, 520),
    apex_y: float = 145.0,
    hoop_pos: tuple = (955, 260),
    total_frames: int = 70,
    fps: float = 30.0,
    width: int = 1280,
    height: int = 720,
):
    """
    Generates a realistic MP4 shooting video with specified parabolic apex and target.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_filename)), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    bg = draw_court_background(width, height, hoop_pos=(hoop_pos[0] + 15, hoop_pos[1]))

    x0, y0 = start_pos
    xt, yt = hoop_pos

    # Parabolic trajectory modeling
    t_end = (total_frames - 8) / fps
    vx = (xt - x0) / t_end

    # Solve for apex timing and parabolic curvature
    ratio = np.sqrt(max(1.0, yt - apex_y) / max(1.0, y0 - apex_y))
    t_apex = t_end / (1.0 + ratio)
    a = (y0 - apex_y) / (t_apex ** 2)

    for f in range(total_frames):
        t = f / fps
        frame = bg.copy()

        if t <= t_end:
            bx = x0 + vx * t
            by = a * ((t - t_apex) ** 2) + apex_y
        else:
            # Ball passing down through the hoop net
            t_post = t - t_end
            bx = xt + 0.15 * vx * t_post
            by = yt + 170 * t_post + 300 * (t_post ** 2)

        draw_realistic_basketball(frame, (bx, by), radius=18)
        out.write(frame)

    out.release()
    print(f"Generated synthetic shot video: {output_filename} ({total_frames} frames, {width}x{height} @ {fps}fps)")


def generate_all_samples(output_dir: str = "."):
    """Creates sample shot 1 (high arc) and sample shot 2 (flatter arc)."""
    shot1_path = os.path.join(output_dir, "shot_sample_1.mp4")
    shot2_path = os.path.join(output_dir, "shot_sample_2.mp4")

    print("Generating CourtVision Sample Video 1 (High Arc, apex_y=140px)...")
    generate_shot_video(
        output_filename=shot1_path,
        start_pos=(160, 520),
        apex_y=140.0,
        hoop_pos=(955, 260),
        total_frames=70,
    )

    print("Generating CourtVision Sample Video 2 (Flatter Arc, apex_y=205px)...")
    generate_shot_video(
        output_filename=shot2_path,
        start_pos=(180, 500),
        apex_y=205.0,
        hoop_pos=(955, 260),
        total_frames=65,
    )

    print("Synthetic videos successfully generated.")
    return shot1_path, shot2_path


if __name__ == "__main__":
    generate_all_samples()
