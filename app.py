"""
app.py
CourtVision – Basketball Shot Analysis System
Streamlit Dashboard Application

A professional sports analytics application for tracking, analyzing,
and comparing basketball shooting trajectories using OpenCV and Computer Vision.
"""

import os
import tempfile
from typing import List, Tuple, Optional, Dict, Any
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from video_processing import VideoProcessor
from ball_tracker import BallTracker, BallDetection
from trajectory_analyzer import TrajectoryAnalyzer, ShotMetrics
from shot_comparator import ShotComparator
from generate_sample_video import generate_all_samples


# ==========================================
# PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="CourtVision – Basketball Shot Analysis",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Sports Analytics Dark Theme CSS
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background-color: #0E1117;
        color: #E0E0E0;
    }
    
    /* Headers & Accent */
    h1, h2, h3 {
        color: #FFFFFF !important;
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    .brand-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #FF6D00, #00E5FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .brand-sub {
        color: #90A4AE;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    
    /* Metric Cards */
    .metric-card {
        background-color: #1A1F2C;
        border: 1px solid #2A3142;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #00E5FF;
        margin: 4px 0;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #90A4AE;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Scientific Disclaimer Box */
    .disclaimer-box {
        background-color: #141B2D;
        border-left: 4px solid #FF9100;
        padding: 14px 18px;
        border-radius: 4px;
        margin: 16px 0;
        font-size: 0.9rem;
        color: #CFD8DC;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        font-weight: 600;
        border-radius: 6px 6px 0 0;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# HELPER FUNCTIONS & CACHED RESOURCES
# ==========================================
@st.cache_resource
def ensure_sample_videos():
    """Generates synthetic sample shot videos if not already present."""
    shot1 = "shot_sample_1.mp4"
    shot2 = "shot_sample_2.mp4"
    if not os.path.exists(shot1) or not os.path.exists(shot2):
        generate_all_samples(".")
    return shot1, shot2


def save_uploaded_file(uploaded_file) -> str:
    """Saves uploaded video file to a temporary file and returns its path."""
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    tfile.close()
    return tfile.name


# ==========================================
# SIDEBAR CONTROLS
# ==========================================
st.sidebar.markdown("## 🏀 **CourtVision Controls**")
nav_mode = st.sidebar.radio(
    "Navigation Mode",
    ["🎯 Single Shot Analysis", "⚖️ Shot Comparison", "📚 CV Methodology Guide"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ **Detection Tuning & Presets**")

footage_profile = st.sidebar.selectbox(
    "🏀 Video Footage Profile",
    ["🌟 Real NBA Broadcast / Fast Action", "🎯 Practice Court / High Contrast", "🛠️ Custom HSV Tuning"],
    index=0,
    help="NBA Broadcast optimizes for compression, motion blur, and dim crowd lighting."
)

if footage_profile == "🌟 Real NBA Broadcast / Fast Action":
    def_h = (5, 26)
    def_s = (35, 255)
    def_v = (30, 245)
    def_circ = 0.28
    def_min_a, def_max_a = 40, 6500
    tracker_mode = "nba_broadcast"
    use_motion = True
elif footage_profile == "🎯 Practice Court / High Contrast":
    def_h = (5, 25)
    def_s = (95, 255)
    def_v = (80, 255)
    def_circ = 0.40
    def_min_a, def_max_a = 50, 12000
    tracker_mode = "practice_court"
    use_motion = False
else:
    def_h = (5, 25)
    def_s = (45, 255)
    def_v = (40, 255)
    def_circ = 0.32
    def_min_a, def_max_a = 40, 10000
    tracker_mode = "custom"
    use_motion = True

with st.sidebar.expander("Fine-Tune HSV & Shape Thresholds", expanded=False):
    st.caption("Customize orange segmentation and circularity filtering for varying court lighting.")
    hue_range = st.slider("Hue (H) Range", 0, 179, def_h)
    sat_range = st.slider("Saturation (S) Range", 0, 255, def_s)
    val_range = st.slider("Value (V) Range", 0, 255, def_v)
    min_circ = st.slider("Min Circularity", 0.1, 1.0, def_circ, 0.02,
                         help="Lower (e.g. 0.28) allows motion-blurred basketballs during fast flight.")
    min_area = st.number_input("Min Contour Area (px)", value=def_min_a, step=10)
    max_area = st.number_input("Max Contour Area (px)", value=def_max_a, step=500)

sample_shot1_path, sample_shot2_path = ensure_sample_videos()


# ==========================================
# MODE 1: SINGLE SHOT ANALYSIS
# ==========================================
if nav_mode == "🎯 Single Shot Analysis":
    st.markdown('<div class="brand-title">CourtVision – Basketball Shot Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-sub">Computer Vision trajectory tracking, multi-frame hoop calibration & telemetry dashboard.</div>', unsafe_allow_html=True)

    # 1. Video Source Selection
    st.markdown("### 1. Video Source")
    # Scan local workspace for available video clips
    local_mp4s = [f for f in os.listdir(".") if f.endswith(".mp4") and not f.startswith("annotated_")]
    video_options = []
    if "ss_4.mp4" in local_mp4s:
        video_options.append("🌟 ss_4.mp4 (Steph Curry NBA Game Clip)")
    if "ss5.mp4" in local_mp4s:
        video_options.append("🌟 ss5.mp4 (NBA Game Clip)")
    video_options.extend(["🎬 Sample Shot 1 (High Arc)", "🎬 Sample Shot 2 (Flatter Arc)", "📁 Upload MP4 Video"])

    video_source_type = st.radio(
        "Choose video source:",
        video_options,
        index=0 if "ss_4.mp4" in local_mp4s else 2,
        horizontal=True
    )

    video_path = None
    if "ss_4.mp4" in video_source_type:
        video_path = "ss_4.mp4"
        st.info("Loaded **ss_4.mp4 (Stephen Curry vs Clippers NBA Clip)**.")
    elif "ss5.mp4" in video_source_type:
        video_path = "ss5.mp4"
        st.info("Loaded **ss5.mp4 (NBA Clip)**.")
    elif "Sample Shot 1" in video_source_type:
        video_path = sample_shot1_path
        st.info("Loaded built-in **Sample Shot 1** (High arc projectile trajectory).")
    elif "Sample Shot 2" in video_source_type:
        video_path = sample_shot2_path
        st.info("Loaded built-in **Sample Shot 2** (Flatter release trajectory).")
    else:
        uploaded_video = st.file_uploader("Upload Basketball Shot MP4 Video", type=["mp4", "mov", "avi"])
        if uploaded_video is not None:
            video_path = save_uploaded_file(uploaded_video)
            st.success(f"Video uploaded: {uploaded_video.name}")
        else:
            st.warning("Please upload an MP4 basketball shooting video or select a video above.")

    if video_path and os.path.exists(video_path):
        try:
            vp = VideoProcessor(video_path)
        except Exception as e:
            st.error(f"Error opening video file: {e}")
            st.stop()

        # Display Video Metadata
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Resolution", f"{vp.width} x {vp.height}")
        col_m2.metric("Frame Rate", f"{vp.fps:.1f} FPS")
        col_m3.metric("Total Frames", f"{vp.total_frames}")
        col_m4.metric("Duration", f"{vp.duration_sec:.2f} s")

        st.markdown("---")

        # 2. Hoop Region Selection & Verification
        st.markdown("### 2. Hoop Target Calibration")
        st.caption("Scrub to the frame where the hoop/rim is visible (e.g. towards the end of the shot clip).")

        is_curry = "ss_4" in str(video_path)

        col_hoop_ctrl, col_hoop_view = st.columns([1, 1.4])

        with col_hoop_ctrl:
            hoop_calib_frame = st.slider(
                "Scrub Video to Locate Hoop (Rim)",
                0, vp.total_frames - 1,
                value=78 if (is_curry and vp.total_frames > 78) else min(int(vp.total_frames * 0.85), vp.total_frames - 1),
                help="Scrub to the frame where the rim and backboard appear clearly in the camera view."
            )

            # Quick Presets
            st.caption("Quick Alignment Presets:")
            col_p1, col_p2, col_p3 = st.columns(3)
            if col_p1.button("⬅️ Left Rim", help="Common in NBA broadcast angle"):
                st.session_state["calib_hx"] = 475 if is_curry else int(vp.width * 0.38)
                st.session_state["calib_hy"] = 495 if is_curry else int(vp.height * 0.85)
            if col_p2.button("➡️ Right Rim", help="Practice court / right side"):
                st.session_state["calib_hx"] = int(vp.width * 0.75)
                st.session_state["calib_hy"] = int(vp.height * 0.36)
            if col_p3.button("🎯 Center"):
                st.session_state["calib_hx"] = int(vp.width * 0.50)
                st.session_state["calib_hy"] = int(vp.height * 0.45)

            cur_hx = st.session_state.get(
                "calib_hx",
                475 if is_curry else (int(vp.width * 0.746) if vp.width == 1280 else int(vp.width * 0.5))
            )
            cur_hy = st.session_state.get(
                "calib_hy",
                495 if is_curry else (int(vp.height * 0.361) if vp.height == 720 else int(vp.height * 0.4))
            )

            hoop_x = st.slider("Hoop Center X (px)", 0, vp.width, min(cur_hx, vp.width), step=2)
            hoop_y = st.slider("Hoop Center Y (px)", 0, vp.height, min(cur_hy, vp.height), step=2)
            hoop_radius = st.slider("Hoop Target Radius (px)", 10, 80, 26, step=2)

            st.caption(f"📍 Selected Hoop Position: `({hoop_x}, {hoop_y})`, Radius: `{hoop_radius} px` (Frame {hoop_calib_frame})")

            preview_hsv = st.checkbox("🔍 Preview Orange HSV Mask on this Frame", value=False)

        with col_hoop_view:
            frame_hoop = vp.get_frame(hoop_calib_frame)
            if frame_hoop is not None:
                preview_frame = frame_hoop.copy()
                # Draw hoop preview
                cv2.circle(preview_frame, (hoop_x, hoop_y), hoop_radius, (255, 50, 50), 3)
                cv2.circle(preview_frame, (hoop_x, hoop_y), 4, (255, 50, 50), -1)
                cv2.line(preview_frame, (hoop_x - 15, hoop_y), (hoop_x + 15, hoop_y), (255, 50, 50), 2)
                cv2.line(preview_frame, (hoop_x, hoop_y - 15), (hoop_x, hoop_y + 15), (255, 50, 50), 2)
                cv2.putText(preview_frame, "HOOP TARGET", (hoop_x - 45, hoop_y - hoop_radius - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 50, 50), 2)

                if preview_hsv:
                    bgr_h = cv2.cvtColor(frame_hoop, cv2.COLOR_RGB2BGR)
                    dummy_tracker = BallTracker(
                        hsv_lower=(hue_range[0], sat_range[0], val_range[0]),
                        hsv_upper=(hue_range[1], sat_range[1], val_range[1]),
                    )
                    mask_h = dummy_tracker.create_color_mask(bgr_h)
                    mask_rgb = cv2.cvtColor(mask_h, cv2.COLOR_GRAY2RGB)
                    st.image(mask_rgb, caption=f"Frame {hoop_calib_frame:03d} HSV Mask (White = Orange)", use_container_width=True)
                else:
                    st.image(preview_frame, caption=f"Frame {hoop_calib_frame:03d}: Rim Target Alignment", use_container_width=True)

        st.markdown("---")

        # 3. Shooter Ball Release Calibration
        st.markdown("### 3. Ball Release Calibration (Locks Tracker to Real Ball)")
        st.caption("Locks the tracking engine directly onto the basketball in the shooter's hands, preventing false locks on defender jerseys or spectators.")

        col_seed_ctrl, col_seed_view = st.columns([1, 1.4])
        with col_seed_ctrl:
            enable_seed = st.checkbox("🎯 Lock Initial Ball Release Position", value=True,
                                      help="Strongly recommended for NBA game clips with crowd spectators and active players.")
            if enable_seed:
                seed_frame_idx = st.slider(
                    "Scrub to Shooter Release Frame",
                    0, min(50, vp.total_frames - 1),
                    value=12 if (is_curry and vp.total_frames > 12) else min(12, vp.total_frames - 1),
                    help="Find the frame where the shooter lifts or releases the ball."
                )
                def_bx = 725 if is_curry else (int(vp.width * 0.57) if vp.width > 1000 else int(vp.width * 0.5))
                def_by = 135 if is_curry else int(vp.height * 0.25)
                seed_bx = st.slider("Ball Center X (px)", 0, vp.width, def_bx, step=2)
                seed_by = st.slider("Ball Center Y (px)", 0, vp.height, def_by, step=2)
                st.caption(f"🏀 Initial Ball Locked At: `({seed_bx}, {seed_by})` on Frame {seed_frame_idx}")
            else:
                seed_frame_idx = 0
                seed_bx, seed_by = None, None

        with col_seed_view:
            if enable_seed:
                frame_seed = vp.get_frame(seed_frame_idx)
                if frame_seed is not None:
                    prev_seed = frame_seed.copy()
                    cv2.circle(prev_seed, (seed_bx, seed_by), 18, (0, 255, 0), 3)
                    cv2.circle(prev_seed, (seed_bx, seed_by), 3, (0, 255, 0), -1)
                    cv2.putText(prev_seed, "BALL RELEASE", (seed_bx - 45, seed_by - 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                    st.image(prev_seed, caption=f"Frame {seed_frame_idx:03d}: Ball Release Lock Preview", use_container_width=True)

        st.markdown("---")

        # 4. Execution Button
        run_analysis = st.button("🚀 Run CourtVision Shot Analysis", type="primary", use_container_width=True)

        analysis_key = f"analysis_{video_path}_{hoop_x}_{hoop_y}_{seed_bx}_{seed_by}"

        if run_analysis:
            progress_bar = st.progress(0, text="Initializing Computer Vision tracker...")

            tracker = BallTracker(
                mode=tracker_mode,
                hsv_lower=(hue_range[0], sat_range[0], val_range[0]),
                hsv_upper=(hue_range[1], sat_range[1], val_range[1]),
                min_area=float(min_area),
                max_area=float(max_area),
                min_circularity=float(min_circ),
                initial_ball_pos=(seed_bx, seed_by) if enable_seed else None,
                initial_frame_idx=seed_frame_idx if enable_seed else 0,
                use_motion_filter=use_motion,
            )

            def update_progress(curr, total):
                pct = min(1.0, max(0.0, curr / total))
                progress_bar.progress(pct, text=f"Tracking basketball trajectory... ({curr}/{total} frames)")

            temp_output_video = os.path.join(tempfile.gettempdir(), f"courtvision_annotated_{os.path.basename(video_path)}")

            detections, annotated_path = vp.process_and_annotate(
                tracker=tracker,
                output_video_path=temp_output_video,
                hoop_center=(hoop_x, hoop_y),
                hoop_radius=float(hoop_radius),
                progress_callback=update_progress,
            )

            progress_bar.progress(1.0, text="Computing trajectory kinematics and visual charts...")

            analyzer = TrajectoryAnalyzer(hoop_center=(hoop_x, hoop_y), hoop_radius=float(hoop_radius))
            metrics = analyzer.analyze(detections)

            st.session_state[analysis_key] = {
                "detections": detections,
                "metrics": metrics,
                "annotated_video": annotated_path,
                "hoop": (hoop_x, hoop_y, hoop_radius),
                "vp": vp,
            }

            st.success("✅ Video Analysis Complete!")

        # 4. Display Results Dashboard
        if analysis_key in st.session_state:
            data = st.session_state[analysis_key]
            metrics: ShotMetrics = data["metrics"]
            detections: List[BallDetection] = data["detections"]
            annotated_video_path = data["annotated_video"]
            vp: VideoProcessor = data["vp"]
            hx, hy, hr = data["hoop"]

            analyzer = TrajectoryAnalyzer(hoop_center=(hx, hy), hoop_radius=hr)

            st.markdown("### 3. Kinematic Metrics & Shot Telemetry")

            col_k1, col_k2, col_k3, col_k4, col_k5, col_k6 = st.columns(6)

            with col_k1:
                val = f"{metrics.launch_angle_deg:.1f}°" if metrics.launch_angle_deg is not None else "N/A"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Launch Angle</div>
                    <div class="metric-value">{val}</div>
                    <div style="font-size: 0.75rem; color: #90A4AE;">Elevation relative to floor</div>
                </div>
                """, unsafe_allow_html=True)

            with col_k2:
                val = f"{metrics.closest_distance_px:.1f} px" if metrics.closest_distance_px is not None else "N/A"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Closest to Hoop</div>
                    <div class="metric-value" style="color: #FF5252;">{val}</div>
                    <div style="font-size: 0.75rem; color: #90A4AE;">Image-plane proximity</div>
                </div>
                """, unsafe_allow_html=True)

            with col_k3:
                val = f"{metrics.apex_point[1]} px" if metrics.apex_point else "N/A"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Apex Height (Y)</div>
                    <div class="metric-value" style="color: #FFD600;">{val}</div>
                    <div style="font-size: 0.75rem; color: #90A4AE;">Peak elevation in frame</div>
                </div>
                """, unsafe_allow_html=True)

            with col_k4:
                val = f"{metrics.flight_duration_sec:.2f} s"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Hang Time</div>
                    <div class="metric-value">{val}</div>
                    <div style="font-size: 0.75rem; color: #90A4AE;">Tracked flight duration</div>
                </div>
                """, unsafe_allow_html=True)

            with col_k5:
                val = f"{metrics.tracking_success_rate:.1f}%"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Tracking Rate</div>
                    <div class="metric-value" style="color: #00E676;">{val}</div>
                    <div style="font-size: 0.75rem; color: #90A4AE;">{metrics.detected_frames_count}/{metrics.total_frames_analyzed} frames</div>
                </div>
                """, unsafe_allow_html=True)

            with col_k6:
                val = f"{metrics.avg_speed_px_per_sec:.0f} px/s" if metrics.avg_speed_px_per_sec else "N/A"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Average Speed</div>
                    <div class="metric-value">{val}</div>
                    <div style="font-size: 0.75rem; color: #90A4AE;">Image-plane velocity</div>
                </div>
                """, unsafe_allow_html=True)

            # Scientific Disclaimer Callout
            st.markdown("""
            <div class="disclaimer-box">
                <strong>⚠️ Sports Analytics & Computer Vision Specification:</strong><br>
                All measurements reported above are strictly calculated in the <strong>2D image plane (pixels)</strong>.
                In single-camera (monocular) sports analysis, proximity to the hoop reflects shot aim and directional alignment;
                however, <strong>monocular image-plane proximity does not prove a successful basket</strong> without multi-camera 3D depth calibration or rim vibration / net sensors.
            </div>
            """, unsafe_allow_html=True)

            # Trajectory Visualizations & Video Playback Tabs
            res_tab1, res_tab2, res_tab3, res_tab4 = st.tabs([
                "📊 2D Trajectory Arc",
                "📈 Kinematics vs Time",
                "🎥 Video Playback & HUD",
                "📋 Trajectory Dataset"
            ])

            with res_tab1:
                st.markdown("#### Spatial Trajectory (Camera Coordinate System)")
                st.caption("Plots the tracked ball coordinates with inverted Y-axis to match camera perspective.")
                fig_traj = analyzer.generate_trajectory_plot(
                    detections=detections,
                    frame_width=vp.width,
                    frame_height=vp.height,
                    metrics=metrics,
                    title="CourtVision – 2D Ball Flight Trajectory"
                )
                st.pyplot(fig_traj)

            with res_tab2:
                st.markdown("#### Position Kinematics & Hoop Distance Curves")
                fig_kin = analyzer.generate_kinematics_plot(
                    detections=detections,
                    metrics=metrics,
                    title="Kinematic Profiles over Time"
                )
                st.pyplot(fig_kin)

            with res_tab3:
                st.markdown("#### Annotated Video Playback")
                col_v1, col_v2 = st.columns([1.4, 1])

                with col_v1:
                    if annotated_video_path and os.path.exists(annotated_video_path):
                        with open(annotated_video_path, "rb") as vf:
                            video_bytes = vf.read()
                        st.video(video_bytes)
                    else:
                        st.info("Video player preview.")

                with col_v2:
                    st.markdown("##### Frame Scrubber Inspector")
                    scrub_idx = st.slider("Inspect Frame Index", 0, vp.total_frames - 1, min(15, vp.total_frames - 1))
                    scrub_frame = vp.get_frame(scrub_idx)
                    if scrub_frame is not None and scrub_idx < len(detections):
                        annotated_scrub = vp.draw_annotations(
                            frame_bgr=cv2.cvtColor(scrub_frame, cv2.COLOR_RGB2BGR),
                            current_det=detections[scrub_idx],
                            tracked_history=detections[:scrub_idx + 1],
                            hoop_center=(hx, hy),
                            hoop_radius=hr,
                            launch_angle=metrics.launch_angle_deg,
                            closest_dist=metrics.closest_distance_px,
                        )
                        st.image(cv2.cvtColor(annotated_scrub, cv2.COLOR_BGR2RGB),
                                 caption=f"Frame {scrub_idx:03d} Overlay", use_container_width=True)

            with res_tab4:
                st.markdown("#### Trajectory Coordinate Table")
                df_traj = vp.export_csv(detections, hoop_center=(hx, hy))
                st.dataframe(df_traj, use_container_width=True)

            # 5. Export Section
            st.markdown("---")
            st.markdown("### 4. Export & Download")
            col_exp1, col_exp2 = st.columns(2)

            with col_exp1:
                df_export = vp.export_csv(detections, hoop_center=(hx, hy))
                csv_data = df_export.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Trajectory CSV Dataset",
                    data=csv_data,
                    file_name="courtvision_trajectory_data.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            with col_exp2:
                if annotated_video_path and os.path.exists(annotated_video_path):
                    with open(annotated_video_path, "rb") as vf:
                        st.download_button(
                            label="📥 Download Annotated MP4 Video",
                            data=vf,
                            file_name="courtvision_annotated_shot.mp4",
                            mime="video/mp4",
                            use_container_width=True
                        )


# ==========================================
# MODE 2: SHOT COMPARISON
# ==========================================
elif nav_mode == "⚖️ Shot Comparison":
    st.markdown('<div class="brand-title">CourtVision – Multi-Shot Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-sub">Compare two shooting videos: trajectory arcs, release angles, apexes, and hoop proximity.</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### 🏀 Shot A Configuration")
        source_a = st.selectbox("Select Shot A Video",
                                ["Sample Shot 1 (High Arc)", "Upload MP4"],
                                key="src_a")
        path_a = sample_shot1_path
        if source_a == "Upload MP4":
            up_a = st.file_uploader("Upload Shot A", type=["mp4", "mov"], key="up_a")
            if up_a:
                path_a = save_uploaded_file(up_a)

        hx_a = st.number_input("Shot A Hoop X", value=955, key="hx_a")
        hy_a = st.number_input("Shot A Hoop Y", value=260, key="hy_a")

    with col_b:
        st.markdown("### 🏀 Shot B Configuration")
        source_b = st.selectbox("Select Shot B Video",
                                ["Sample Shot 2 (Flatter Arc)", "Upload MP4"],
                                key="src_b")
        path_b = sample_shot2_path
        if source_b == "Upload MP4":
            up_b = st.file_uploader("Upload Shot B", type=["mp4", "mov"], key="up_b")
            if up_b:
                path_b = save_uploaded_file(up_b)

        hx_b = st.number_input("Shot B Hoop X", value=955, key="hx_b")
        hy_b = st.number_input("Shot B Hoop Y", value=260, key="hy_b")

    st.markdown("---")
    compare_btn = st.button("⚖️ Compare Shot Trajectories", type="primary", use_container_width=True)

    if compare_btn:
        with st.spinner("Processing Shot A and Shot B trajectories..."):
            # Tracker A
            t_a = BallTracker(
                mode=tracker_mode,
                hsv_lower=(hue_range[0], sat_range[0], val_range[0]),
                hsv_upper=(hue_range[1], sat_range[1], val_range[1]),
                min_circularity=float(min_circ),
            )
            vp_a = VideoProcessor(path_a)
            dets_a, _ = vp_a.process_and_annotate(t_a, hoop_center=(hx_a, hy_a))
            m_a = TrajectoryAnalyzer(hoop_center=(hx_a, hy_a)).analyze(dets_a)

            # Tracker B
            t_b = BallTracker(
                mode=tracker_mode,
                hsv_lower=(hue_range[0], sat_range[0], val_range[0]),
                hsv_upper=(hue_range[1], sat_range[1], val_range[1]),
                min_circularity=float(min_circ),
            )
            vp_b = VideoProcessor(path_b)
            dets_b, _ = vp_b.process_and_annotate(t_b, hoop_center=(hx_b, hy_b))
            m_b = TrajectoryAnalyzer(hoop_center=(hx_b, hy_b)).analyze(dets_b)

            st.session_state["comparison_data"] = {
                "m_a": m_a, "dets_a": dets_a, "vp_a": vp_a,
                "m_b": m_b, "dets_b": dets_b, "vp_b": vp_b,
            }

    if "comparison_data" in st.session_state:
        cdata = st.session_state["comparison_data"]
        m_a: ShotMetrics = cdata["m_a"]
        m_b: ShotMetrics = cdata["m_b"]
        dets_a = cdata["dets_a"]
        dets_b = cdata["dets_b"]
        vp_a: VideoProcessor = cdata["vp_a"]

        # Comparison Key Stats
        st.markdown("### 📊 Side-by-Side Kinematics")
        col_c1, col_c2, col_c3 = st.columns(3)

        with col_c1:
            ang_a = f"{m_a.launch_angle_deg:.1f}°" if m_a.launch_angle_deg else "N/A"
            ang_b = f"{m_b.launch_angle_deg:.1f}°" if m_b.launch_angle_deg else "N/A"
            st.metric("Launch Elevation Angle", f"A: {ang_a}", delta=f"B: {ang_b}")

        with col_c2:
            dist_a = f"{m_a.closest_distance_px:.1f} px" if m_a.closest_distance_px else "N/A"
            dist_b = f"{m_b.closest_distance_px:.1f} px" if m_b.closest_distance_px else "N/A"
            st.metric("Closest Approach to Hoop", f"A: {dist_a}", delta=f"B: {dist_b}")

        with col_c3:
            st.metric("Flight Hang Time", f"A: {m_a.flight_duration_sec:.2f} s",
                      delta=f"B: {m_b.flight_duration_sec:.2f} s")

        # Visual Overlay Plot
        st.markdown("### 📈 Trajectory Arc Comparison Overlay")
        fig_comp = ShotComparator.generate_comparison_plot(
            dets_a=dets_a, metrics_a=m_a, name_a="Shot A",
            dets_b=dets_b, metrics_b=m_b, name_b="Shot B",
            frame_width=vp_a.width, frame_height=vp_a.height
        )
        st.pyplot(fig_comp)

        # Tabular Comparison
        st.markdown("### 📋 Metric Comparison Matrix")
        df_comp = ShotComparator.build_comparison_dataframe("Shot A", m_a, "Shot B", m_b)
        st.dataframe(df_comp, use_container_width=True)

        # Analytical Commentary
        st.markdown("### 🧠 Kinematic Comparison Observations")
        insights = ShotComparator.generate_analytical_insights(m_a, "Shot A", m_b, "Shot B")
        for item in insights:
            st.markdown(f"- {item}")


# ==========================================
# MODE 3: CV METHODOLOGY & EDUCATIONAL GUIDE
# ==========================================
elif nav_mode == "📚 CV Methodology Guide":
    st.markdown('<div class="brand-title">CourtVision – Computer Vision Practical Guide</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-sub">Comprehensive documentation of image processing techniques, mathematical formulations, and engineering principles.</div>', unsafe_allow_html=True)

    st.markdown(r"""
    ### 1. Color Space Transformation (BGR to HSV)
    Standard RGB/BGR color channels combine luminance (lighting intensity) and chrominance (color information), making segmentation highly vulnerable to arena lighting, glare, and court reflections.
    The **HSV (Hue, Saturation, Value)** cylindrical color model decouples chromaticity:
    - **Hue ($H \in [0, 179]$ in OpenCV):** Encodes the pure color wavelength. Basketball orange is predominantly concentrated between $5^{\circ}$ and $25^{\circ}$.
    - **Saturation ($S \in [0, 255]$):** Isolates the purity or vibrancy of the color, distinguishing the deep leather orange ball from pale wooden floorboards.
    - **Value ($V \in [0, 255]$):** Represents brightness, allowing the tracker to handle dynamic shadows.

    ```python
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([5, 95, 80]), np.array([25, 255, 255]))
    ```

    ---

    ### 2. Morphological Operations
    Raw color masks often suffer from salt-and-pepper noise and interior gaps created by black ball seam lines.
    - **Morphological Opening ($A \circ B = (A \ominus B) \oplus B$):** Erosion followed by dilation to eliminate detached background noise.
    - **Morphological Closing ($A \bullet B = (A \oplus B) \ominus B$):** Dilation followed by erosion to bridge disconnected seam lines and form a solid spherical blob.

    ---

    ### 3. Shape Discrimination via Circularity Quotient
    To reject false orange detections (such as referee uniforms, court paint, or player shoes), every contour is evaluated using the isoperimetric circularity ratio:
    $$\\text{Circularity} = \\frac{4 \\pi \\cdot \\text{Area}}{\\text{Perimeter}^2}$$
    - For an ideal circle: $\\text{Circularity} = 1.0$.
    - For a square: $\\text{Circularity} \\approx 0.785$.
    - For irregular or elongated blobs: $\\text{Circularity} < 0.40$.

    ---

    ### 4. Centroid Localization & Motion Continuity Tracking
    The basketball's spatial center is determined using zero-th and first-order image moments:
    $$C_x = \\frac{M_{10}}{M_{00}}, \\quad C_y = \\frac{M_{01}}{M_{00}}$$
    Temporal gating projects the ball's expected location in frame $t+1$ based on its moving velocity:
    $$\\vec{P}_{\\text{pred}} = \\vec{P}_t + \\vec{v}_t$$
    Candidates outside a bounding distance gate $\\Delta d > d_{\\max}$ are discarded to prevent erratic spatial jumps.

    ---

    ### 5. Kinematics & Launch Angle Estimation
    In monocular video coordinates, the origin $(0,0)$ resides at the top-left, meaning vertical coordinate $Y$ increases downward.
    Upward physical displacement is computed as $-\\Delta Y$.
    Using multi-frame vector regression over the initial release segment:
    $$\\theta_{\\text{launch}} = \\arctan2(-\\Delta Y, |\\Delta X|) \\times \\frac{180}{\\pi}$$

    ---

    ### 6. Critical Distinction: 2D Image Plane vs 3D Physical Space
    > **⚠️ Key Principle in Sports Vision Analytics:**
    > Monocular 2D video projects a three-dimensional physical arc onto a two-dimensional sensor plane.
    > An image-plane distance of $15$ pixels between the ball centroid and the rim center demonstrates directional trajectory alignment; however, **it does not guarantee that the ball passed inside the hoop rim in 3D space (depth dimension $Z$)**.
    > Rigorous sports analytics must never claim a shot scored based solely on monocular 2D proximity without stereo cameras or physical sensors.
    """)
