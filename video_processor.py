"""
Video processing for cell compression analysis.
Detects cell boundaries and validates compression alignment.
"""

import cv2
import numpy as np
from pathlib import Path
import tempfile
import os


class VideoProcessor:
    """
    Process AFM compression videos to extract cell information
    and validate compression quality.
    """

    def __init__(self, video_path):
        """
        Initialize video processor.

        Parameters:
        -----------
        video_path : str
            Path to video file (.wmv, .mp4, .avi)
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.frames = []
        self.analysis = {}

    def extract_frames(self, start_frame=0, end_frame=None, step=1):
        """
        Extract frames from video.

        Parameters:
        -----------
        start_frame : int
            Starting frame index
        end_frame : int, optional
            Ending frame index
        step : int
            Frame step (e.g., step=5 gets every 5th frame)

        Returns:
        --------
        list of numpy arrays
        """
        if end_frame is None:
            end_frame = self.total_frames

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        self.frames = []

        frame_idx = start_frame
        while frame_idx < end_frame:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.frames.append(frame_rgb)

            frame_idx += step
            # Move to next frame to extract
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        return self.frames

    def detect_cell_boundary(self, frame, blur_kernel=(5, 5), threshold=50):
        """
        Detect cell boundary using edge detection.

        Parameters:
        -----------
        frame : numpy array
            Video frame (RGB)
        blur_kernel : tuple
            Kernel size for Gaussian blur
        threshold : int
            Threshold for edge detection

        Returns:
        --------
        dict with cell detection results
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, blur_kernel, 0)

        # Edge detection
        edges = cv2.Canny(blurred, threshold, threshold * 2)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return {'detected': False, 'error': 'No cell boundary detected'}

        # Get largest contour (likely the cell)
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        # Fit circle/ellipse
        if area > 50:  # Minimum area check
            # Fit ellipse
            if len(largest_contour) >= 5:
                ellipse = cv2.fitEllipse(largest_contour)
                center, (width, height), angle = ellipse

                # Fit circle
                (cx, cy), radius = cv2.minEnclosingCircle(largest_contour)

                return {
                    'detected': True,
                    'contour': largest_contour,
                    'area': area,
                    'ellipse': ellipse,
                    'ellipse_center': center,
                    'ellipse_axes': (width, height),
                    'circle_center': (cx, cy),
                    'circle_radius': radius
                }
        else:
            return {'detected': False, 'error': 'Cell area too small'}

    def analyze_compression_alignment(self, frames=None, sample_rate=None):
        """
        Analyze if compression is "head-on" (centered) vs off-axis.

        Parameters:
        -----------
        frames : list, optional
            List of frames to analyze. If None, uses extracted frames.
        sample_rate : int, optional
            Analyze every nth frame

        Returns:
        --------
        dict with alignment analysis
        """
        if frames is None:
            frames = self.frames

        if not frames:
            return {'error': 'No frames available'}

        if sample_rate is None:
            sample_rate = max(1, len(frames) // 10)  # Sample ~10 frames

        alignment_scores = []
        centers = []
        radii = []

        for i, frame in enumerate(frames[::sample_rate]):
            detection = self.detect_cell_boundary(frame)

            if detection['detected']:
                center = detection['circle_center']
                radius = detection['circle_radius']
                centers.append(center)
                radii.append(radius)

                # Calculate frame center
                frame_center = (frame.shape[1] / 2, frame.shape[0] / 2)

                # Distance from frame center (normalized by radius)
                dist = np.sqrt(
                    (center[0] - frame_center[0])**2 +
                    (center[1] - frame_center[1])**2
                )
                normalization = np.sqrt(frame.shape[0]**2 + frame.shape[1]**2) / 2

                # Alignment score: 1.0 = perfectly centered, 0.0 = off-center
                alignment_score = max(0, 1 - (dist / normalization))
                alignment_scores.append(alignment_score)

        if not alignment_scores:
            return {'error': 'Could not detect cell in any frames'}

        alignment_scores = np.array(alignment_scores)
        radii = np.array(radii)

        # Analyze deformation
        deformation_symmetry = 1 - (np.std(radii) / np.mean(radii))

        self.analysis['alignment'] = {
            'mean_alignment_score': float(np.mean(alignment_scores)),
            'alignment_consistency': float(np.std(alignment_scores)),
            'is_head_on': float(np.mean(alignment_scores)) > 0.7,
            'deformation_symmetry': float(max(0, deformation_symmetry)),
            'quality_assessment': self._assess_quality(
                np.mean(alignment_scores),
                deformation_symmetry
            ),
            'frames_analyzed': len(alignment_scores)
        }

        return self.analysis['alignment']

    def _assess_quality(self, alignment, symmetry):
        """
        Assess compression quality.

        Parameters:
        -----------
        alignment : float
            Mean alignment score (0-1)
        symmetry : float
            Deformation symmetry (0-1)

        Returns:
        --------
        str with quality assessment
        """
        quality_score = (alignment + symmetry) / 2

        if quality_score > 0.8:
            return 'EXCELLENT - Head-on compression with symmetric deformation'
        elif quality_score > 0.6:
            return 'GOOD - Acceptable compression quality'
        elif quality_score > 0.4:
            return 'FAIR - Some off-axis compression detected'
        else:
            return 'POOR - Significant off-axis compression or asymmetric deformation'

    def get_frame_images(self, n_frames=5):
        """
        Get representative frames for visualization.

        Parameters:
        -----------
        n_frames : int
            Number of frames to extract

        Returns:
        --------
        list of frames evenly spaced through video
        """
        if not self.frames:
            step = max(1, self.total_frames // n_frames)
            self.extract_frames(step=step)

        return self.frames[:n_frames]

    def close(self):
        """Release video capture."""
        self.cap.release()

    def __del__(self):
        """Cleanup on deletion."""
        self.close()
