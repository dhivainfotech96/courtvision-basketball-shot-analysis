"""
test_courtvision.py
Automated Verification & Unit Tests for CourtVision modules.

Verifies:
1. Video loading and property inspection
2. Ball detection (HSV + circularity) and tracking accuracy
3. Kinematics calculations (Launch angle, Apex, Hoop distance)
4. Video annotation rendering & MP4 output
5. CSV export validation
6. Shot comparison module
"""

import os
import unittest
import numpy as np
import pandas as pd
from video_processing import VideoProcessor
from ball_tracker import BallTracker
from trajectory_analyzer import TrajectoryAnalyzer
from shot_comparator import ShotComparator
from generate_sample_video import generate_all_samples


class TestCourtVision(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Ensure sample videos exist
        cls.shot1_path = "shot_sample_1.mp4"
        cls.shot2_path = "shot_sample_2.mp4"
        if not os.path.exists(cls.shot1_path) or not os.path.exists(cls.shot2_path):
            generate_all_samples(".")

    def test_01_video_inspection(self):
        vp = VideoProcessor(self.shot1_path)
        self.assertEqual(vp.width, 1280)
        self.assertEqual(vp.height, 720)
        self.assertGreater(vp.total_frames, 50)
        self.assertAlmostEqual(vp.fps, 30.0, delta=1.0)

        # Test frame extraction
        frame = vp.get_frame(0)
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (720, 1280, 3))

    def test_02_ball_tracking_accuracy(self):
        tracker = BallTracker()
        vp = VideoProcessor(self.shot1_path)

        detections, _ = vp.process_and_annotate(
            tracker=tracker,
            output_video_path=None,
            hoop_center=(955, 260),
        )

        detected_count = sum(1 for d in detections if d.detected)
        detection_rate = detected_count / len(detections) * 100.0

        # Should achieve high tracking rate on the synthetic video (> 85%)
        self.assertGreater(detection_rate, 85.0, f"Tracking rate was only {detection_rate:.1f}%")

    def test_03_trajectory_kinematics(self):
        tracker = BallTracker()
        vp = VideoProcessor(self.shot1_path)
        hoop_center = (955, 260)

        detections, _ = vp.process_and_annotate(tracker, hoop_center=hoop_center)

        analyzer = TrajectoryAnalyzer(hoop_center=hoop_center, hoop_radius=25.0)
        metrics = analyzer.analyze(detections)

        # Check launch angle (high arc should be between 40 and 65 degrees)
        self.assertIsNotNone(metrics.launch_angle_deg)
        self.assertGreater(metrics.launch_angle_deg, 35.0)
        self.assertLess(metrics.launch_angle_deg, 70.0)

        # Check apex
        self.assertIsNotNone(metrics.apex_point)
        # Apex Y should be well above release Y (lower Y value in image coordinates)
        self.assertLess(metrics.apex_point[1], metrics.release_point[1])

        # Check closest distance to hoop
        self.assertIsNotNone(metrics.closest_distance_px)
        self.assertLess(metrics.closest_distance_px, 60.0)

        # Check plot generation
        fig1 = analyzer.generate_trajectory_plot(detections, vp.width, vp.height, metrics)
        self.assertIsNotNone(fig1)

        fig2 = analyzer.generate_kinematics_plot(detections, metrics)
        self.assertIsNotNone(fig2)

    def test_04_annotated_video_and_csv_export(self):
        tracker = BallTracker()
        vp = VideoProcessor(self.shot1_path)
        output_mp4 = "annotated_test_output.mp4"
        output_csv = "trajectory_test_output.csv"

        if os.path.exists(output_mp4):
            os.remove(output_mp4)
        if os.path.exists(output_csv):
            os.remove(output_csv)

        detections, saved_video = vp.process_and_annotate(
            tracker=tracker,
            output_video_path=output_mp4,
            hoop_center=(955, 260),
        )

        self.assertTrue(os.path.exists(output_mp4))
        self.assertGreater(os.path.getsize(output_mp4), 1000)

        df = vp.export_csv(detections, hoop_center=(955, 260), output_csv_path=output_csv)
        self.assertTrue(os.path.exists(output_csv))
        self.assertIn("frame_index", df.columns)
        self.assertIn("distance_to_hoop_px", df.columns)
        self.assertEqual(len(df), vp.total_frames)

        # Clean up test output artifacts
        try:
            os.remove(output_mp4)
            os.remove(output_csv)
        except OSError:
            pass

    def test_05_shot_comparison(self):
        hoop = (955, 260)
        analyzer = TrajectoryAnalyzer(hoop_center=hoop)

        # Analyze Shot 1
        t1 = BallTracker()
        vp1 = VideoProcessor(self.shot1_path)
        d1, _ = vp1.process_and_annotate(t1, hoop_center=hoop)
        m1 = analyzer.analyze(d1)

        # Analyze Shot 2
        t2 = BallTracker()
        vp2 = VideoProcessor(self.shot2_path)
        d2, _ = vp2.process_and_annotate(t2, hoop_center=hoop)
        m2 = analyzer.analyze(d2)

        # Build comparison table
        df_comp = ShotComparator.build_comparison_dataframe("Shot 1", m1, "Shot 2", m2)
        self.assertIsInstance(df_comp, pd.DataFrame)
        self.assertGreater(len(df_comp), 4)

        # Compare insights
        insights = ShotComparator.generate_analytical_insights(m1, "Shot 1", m2, "Shot 2")
        self.assertIsInstance(insights, list)
        self.assertGreater(len(insights), 2)

        # Compare plot
        fig_comp = ShotComparator.generate_comparison_plot(d1, m1, "Shot 1", d2, m2, "Shot 2")
        self.assertIsNotNone(fig_comp)


if __name__ == "__main__":
    unittest.main()
