# CourtVision – Basketball Shot Analysis System 🏀

An innovative sports analytics and computer vision application that tracks, analyzes, and compares basketball shooting trajectories from video footage using **Python**, **OpenCV**, **NumPy**, **Matplotlib**, and **Streamlit**.

Designed to be mathematically rigorous, visually intuitive, and ideal for a **College Computer Vision & Image Processing** practical examination or academic project.

---

## 🌟 Key Features

1. **Video Upload & Sample Clips**
   - Supports uploading user MP4/MOV basketball shooting videos.
   - Built-in synthetic basketball shot video generator (`shot_sample_1.mp4` and `shot_sample_2.mp4`) for immediate testing and demonstration.
2. **Interactive Hoop Selection**
   - Real-time coordinate calibration ($X$, $Y$) and radius slider with live visual crosshair and ring overlay on the initial frame.
3. **HSV Basketball Color Segmentation**
   - Tunable HSV thresholds for isolating orange basketballs in various arena lighting conditions.
   - Live color mask preview to visualize segmented pixels before running tracking.
4. **Morphological Filtering & Shape Discrimination**
   - Morphological Opening to eliminate salt-and-pepper noise.
   - Morphological Closing to bridge internal seam lines and reflections.
   - Isoperimetric Circularity filter ($\frac{4\pi A}{P^2} \ge 0.40$) and bounding-box aspect ratio constraints to discard false positives (e.g., jerseys, court markings, sneakers).
5. **Ball Tracking with Motion Continuity**
   - Zero-th and first-order image moments for centroid calculation.
   - Velocity-guided temporal distance gating to prevent erratic tracking jumps across frames.
6. **Trajectory Visualization**
   - Dynamic glowing polyline trail and ball target HUD overlaid directly onto video frames.
   - 2D spatial trajectory chart with inverted $Y$-axis matching camera perspective, highlighting Release Point, Apex, and Hoop target.
   - Kinematic time-series graphs: horizontal $X(t)$, vertical $Y(t)$, and hoop distance $d(t)$.
7. **Kinematic Shot Analysis**
   - Initial launch/release elevation angle estimation ($\theta$) via multi-frame regression.
   - Trajectory apex detection (maximum elevation / minimum $Y$ in image frame).
   - Image-plane Euclidean distance to the hoop ($d_{\min}$ in pixels) and relative position vector $(\Delta x, \Delta y)$.
   - Tracking reliability rate (%) and average image-plane velocity (px/s).
8. **Dual-Shot Comparison Mode**
   - Side-by-side analysis of two shooting attempts (Shot A vs Shot B).
   - Dual trajectory arc overlay chart (Shot A in Cyan, Shot B in Orange).
   - Comprehensive metric comparison matrix and objective kinematic insights (arc steepness, apex height, hang time, hoop proximity).
9. **Export & Download**
   - Download annotated MP4 video with HUD telemetry overlays.
   - Download complete frame-by-frame CSV dataset (`frame_index`, `timestamp_sec`, `detected`, `center_x_px`, `center_y_px`, `radius_px`, `distance_to_hoop_px`).
10. **Scientific Integrity & Graceful Error Handling**
    - Strictly reports measurements in 2D image-plane pixels without fabricating real-world units or false basket-scoring claims.
    - Gracefully handles missing files, codec limitations, and missed detections.

---

## 📐 Computer Vision & Mathematical Formulations

### 1. HSV Color Space Segmentation
RGB channels couple brightness with chromaticity. In contrast, the cylindrical **HSV** space decouples color wavelength (**Hue**) from color purity (**Saturation**) and intensity (**Value**):
```
Hue (H):        [5, 25]      (Orange spectrum in OpenCV: 0-179)
Saturation (S): [95, 255]    (Rejects pale wooden floors & white lines)
Value (V):      [80, 255]    (Rejects dark shadows)
```

### 2. Morphological Seam Sealing
$$A \circ B = (A \ominus B) \oplus B \quad \text{(Opening: removes background noise)}$$
$$A \bullet B = (A \oplus B) \ominus B \quad \text{(Closing: seals black ball seam lines)}$$

### 3. Circularity Quotient
Every candidate contour is evaluated against the isoperimetric quotient:
$$\text{Circularity} = \frac{4 \pi \cdot \text{Area}}{\text{Perimeter}^2}$$
- Ideal circle: $1.0$
- Square: $\approx 0.785$
- Irregular or elongated blobs: $< 0.40$ (filtered out)

### 4. Centroid Localization via Image Moments
$$C_x = \frac{M_{10}}{M_{00}}, \quad C_y = \frac{M_{01}}{M_{00}}$$
where $M_{pq} = \sum_x \sum_y x^p y^q I(x, y)$.

### 5. Launch Angle Estimation
In computer vision, the coordinate origin $(0,0)$ is top-left, meaning vertical coordinate $Y$ increases downward. Upward displacement is $-\Delta Y$.
Given initial horizontal displacement $\Delta X$ and upward displacement $-\Delta Y$:
$$\theta_{\text{launch}} = \arctan2(-\Delta Y, |\Delta X|) \times \frac{180}{\pi}$$

### 6. ⚠️ Critical Distinction: 2D Image Plane vs 3D Space
In monocular (single-camera) sports analytics, distance between ball centroid and hoop center is measured in **2D image-plane pixels**:
$$d(t) = \sqrt{(x(t) - H_x)^2 + (y(t) - H_y)^2} \quad \text{[pixels]}$$
> **Important:** While 2D proximity indicates shot aim and trajectory alignment, **2D image proximity does NOT prove that the ball entered the hoop in 3D depth**. Without multi-camera stereo calibration or rim vibration sensors, sports analytics must not falsely declare a successful basket.

---

## 📁 Project Structure

```
compvision/
├── app.py                     # Interactive Streamlit sports analytics dashboard
├── ball_tracker.py            # HSV segmentation, circularity filter, and tracking logic
├── trajectory_analyzer.py     # Kinematics, launch angle, apex, and Matplotlib plots
├── video_processing.py        # Video I/O, HUD overlay rendering, and CSV exporter
├── shot_comparator.py         # Multi-shot comparison table, overlay chart, and insights
├── generate_sample_video.py   # Standalone synthetic realistic basketball shot generator
├── test_courtvision.py        # Automated unit and integration test suite
├── requirements.txt           # Project dependencies
├── shot_sample_1.mp4          # Synthetic test video 1 (High arc shot, ~52° release)
├── shot_sample_2.mp4          # Synthetic test video 2 (Flatter arc shot, ~40° release)
└── README.md                  # Comprehensive documentation and practical guide
```

---

## 💻 Installation & Setup

### Prerequisites
- Python 3.9 or higher (tested and verified on Python 3.13 on Windows).

### 1. Open Terminal / PowerShell
Navigate to the project folder:
```powershell
cd c:\Users\dhiva\OneDrive\Documents\compvision
```

### 2. Install Required Dependencies
```powershell
pip install -r requirements.txt
```
*(Dependencies: `opencv-python`, `numpy`, `matplotlib`, `streamlit`, `pandas`, `pillow`)*

---

## 🚀 How to Run the Application

### Option A: Launch the Streamlit Dashboard (Main Application)
Run the following command in your terminal:
```powershell
streamlit run app.py
```
This will automatically open the dashboard in your default browser at `http://localhost:8501`.

### Option B: Run Automated Verification Tests
Run the comprehensive unit test suite:
```powershell
python test_courtvision.py
```
*Expected output: `Ran 5 tests in ~6s. OK`*

### Option C: Re-generate Synthetic Sample Shooting Videos
To re-generate the high-definition test MP4 clips:
```powershell
python generate_sample_video.py
```

---

## 🎯 User Guide: Step-by-Step

### 1. Single Shot Analysis
1. Select **"🎯 Single Shot Analysis"** in the sidebar navigation.
2. Choose **Sample Shot 1**, **Sample Shot 2**, or upload your own MP4 video.
3. In **Hoop Target Calibration**, adjust the Hoop Center $X$, $Y$ and radius sliders. Notice how the visual ring updates instantly on the preview image.
4. (Optional) Check **"Preview Orange HSV Mask"** to verify color segmentation.
5. Click **"🚀 Run CourtVision Shot Analysis"**.
6. Review the kinematic KPI cards:
   - **Launch Angle (°)**: Calculated elevation relative to the horizontal floor.
   - **Closest to Hoop (px)**: Closest 2D image-plane approach distance.
   - **Apex Height (px)**: Trajectory peak coordinate in the frame.
   - **Hang Time (s)**: Tracked flight duration.
   - **Tracking Reliability (%)**: Fraction of frames with successful detection.
7. Inspect the interactive tabs:
   - **2D Trajectory Arc**: Visual trajectory plot with camera perspective.
   - **Kinematics vs Time**: $X(t)$, $Y(t)$, and hoop distance curves.
   - **Video Playback & HUD**: Annotated video player and frame scrubber.
   - **Trajectory Dataset**: Data table with frame-by-frame coordinates.
8. Click **Download Annotated MP4 Video** or **Download Trajectory CSV Dataset** to save results.

### 2. Multi-Shot Comparison
1. Select **"⚖️ Shot Comparison"** in the sidebar navigation.
2. Select or upload **Shot A** and **Shot B**.
3. Set the hoop target coordinates for each shot.
4. Click **"⚖️ Compare Shot Trajectories"**.
5. Examine the side-by-side metric comparison table, dual-arc overlay chart, and objective kinematic observations.

### 3. CV Methodology Guide
Select **"📚 CV Methodology Guide"** in the sidebar to review the theoretical foundations of color spaces, morphology, circularity, moments, and image-plane kinematics.

---

## 🎓 College Practical Viva Q&A

**Q1: Why use HSV instead of RGB for basketball tracking?**  
*A:* RGB bundles illumination with color chrominance. Under court shadows or arena spotlights, RGB values change drastically. HSV separates Hue (pure color) from Saturation (purity) and Value (brightness), making color thresholding robust across lighting shifts.

**Q2: How does the circularity filter prevent false positives?**  
*A:* Orange jerseys, sneakers, or court lines may share the basketball's color. The isoperimetric quotient $C = \frac{4\pi A}{P^2}$ tests the geometric compactness of the contour. Only shapes with $C \ge 0.40$ (approximating circles) are accepted.

**Q3: Why is $Y$ inverted in image coordinates?**  
*A:* Digital image arrays index rows starting from $0$ at the top down to image height $H$ at the bottom. As a basketball flies upward physically, its image row index $Y$ decreases. The trajectory peak (apex) corresponds to $\min(Y)$.

**Q4: Can this 2D system tell if a shot scored?**  
*A:* No. A single camera projects 3D space onto a 2D plane. The ball passing directly in front of or behind the rim at rim height will appear to overlap the hoop in 2D pixels even if it missed completely in depth ($Z$-axis). Accurate score confirmation requires multi-camera stereo vision or physical sensors.

---

## 📜 License
Developed for academic, research, and computer vision practical demonstration.
