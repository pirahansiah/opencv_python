"""Advanced feature detection, matching, and panorama stitching."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class MatchResult:
    """Feature matching result."""

    keypoints_src: list = field(default_factory=list)
    keypoints_dst: list = field(default_factory=list)
    matches: list = field(default_factory=list)
    good_matches: list = field(default_factory=list)
    homography_matrix: np.ndarray | None = None


class FeatureExtractor:
    """Feature detection, matching, and homography estimation.

    Supports SIFT, ORB, and AKAZE feature detectors with BFMatcher and FLANN matching.
    """

    def __init__(self) -> None:
        self._detectors: dict[str, cv2.Feature2D] = {}
        self._descriptors: dict[str, cv2.DescriptorMatcher] = {}

    def _get_detector(self, method: str) -> cv2.Feature2D:
        """Get or create a feature detector by name."""
        method = method.lower()
        if method in self._detectors:
            return self._detectors[method]

        if method == "sift":
            det = cv2.SIFT_create()
        elif method == "orb":
            det = cv2.ORB_create(nfeatures=5000)
        elif method == "akaze":
            det = cv2.AKAZE_create()
        else:
            raise ValueError(f"Unknown detector: {method}. Use 'sift', 'orb', or 'akaze'.")

        self._detectors[method] = det
        return det

    def extract_sift(
        self, image: np.ndarray
    ) -> tuple[list[cv2.KeyPoint], np.ndarray]:
        """Extract SIFT features.

        Args:
            image: Input BGR or grayscale image.

        Returns:
            Tuple of (keypoints, descriptors).
        """
        gray = self._to_gray(image)
        detector = self._get_detector("sift")
        keypoints, descriptors = detector.detectAndCompute(gray, None)
        return list(keypoints), descriptors

    def extract_orb(
        self, image: np.ndarray, n_features: int = 5000
    ) -> tuple[list[cv2.KeyPoint], np.ndarray]:
        """Extract ORB features.

        Args:
            image: Input BGR or grayscale image.
            n_features: Maximum features to detect.

        Returns:
            Tuple of (keypoints, descriptors).
        """
        gray = self._to_gray(image)
        detector = cv2.ORB_create(nfeatures=n_features)
        keypoints, descriptors = detector.detectAndCompute(gray, None)
        return list(keypoints), descriptors

    def extract_akaze(
        self, image: np.ndarray
    ) -> tuple[list[cv2.KeyPoint], np.ndarray]:
        """Extract AKAZE features.

        Args:
            image: Input BGR or grayscale image.

        Returns:
            Tuple of (keypoints, descriptors).
        """
        gray = self._to_gray(image)
        detector = self._get_detector("akaze")
        keypoints, descriptors = detector.detectAndCompute(gray, None)
        return list(keypoints), descriptors

    def extract(
        self,
        image: np.ndarray,
        method: str = "sift",
    ) -> tuple[list[cv2.KeyPoint], np.ndarray]:
        """Extract features using specified method.

        Args:
            image: Input image.
            method: 'sift', 'orb', or 'akaze'.

        Returns:
            Tuple of (keypoints, descriptors).
        """
        gray = self._to_gray(image)
        detector = self._get_detector(method)
        keypoints, descriptors = detector.detectAndCompute(gray, None)
        return list(keypoints), descriptors

    def match_features(
        self,
        desc1: np.ndarray,
        desc2: np.ndarray,
        method: str = "bf",
        ratio_test: float = 0.75,
        detector_name: str = "sift",
    ) -> list[cv2.DMatch]:
        """Match features between two descriptor sets.

        Args:
            desc1: Source descriptors.
            desc2: Destination descriptors.
            method: 'bf' (BFMatcher) or 'flann' (FLANN-based).
            ratio_test: Lowe's ratio test threshold.
            detector_name: Detector name (affects norm selection for BFMatcher).

        Returns:
            List of good matches after ratio test.
        """
        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
            return []

        if method == "bf":
            norm = cv2.NORM_L2 if detector_name.lower() != "orb" else cv2.NORM_HAMMING
            matcher = cv2.BFMatcher(norm)
        elif method == "flann":
            if detector_name.lower() in ("orb",):
                index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
            else:
                index_params = dict(algorithm=1, trees=5)
            search_params = dict(checks=50)
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            raise ValueError(f"Unknown matcher: {method}. Use 'bf' or 'flann'.")

        try:
            knn_matches = matcher.knnMatch(desc1, desc2, k=2)
        except cv2.error:
            return []

        good: list[cv2.DMatch] = []
        for pair in knn_matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < ratio_test * n.distance:
                    good.append(m)

        return good

    def homography(
        self,
        keypoints1: list[cv2.KeyPoint],
        keypoints2: list[cv2.KeyPoint],
        matches: list[cv2.DMatch],
        ransac_threshold: float = 5.0,
    ) -> np.ndarray | None:
        """Compute homography matrix from matched keypoints.

        Args:
            keypoints1: Source keypoints.
            keypoints2: Destination keypoints.
            matches: Matched features.
            ransac_threshold: RANSAC reprojection threshold.

        Returns:
            3x3 homography matrix or None if insufficient matches.
        """
        if len(matches) < 4:
            return None

        pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransac_threshold)
        return H

    def match_images(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        method: str = "sift",
        matcher: str = "flann",
        ratio_test: float = 0.75,
    ) -> MatchResult:
        """Full feature matching pipeline between two images.

        Args:
            img1: Source image.
            img2: Destination image.
            method: Feature detector ('sift', 'orb', 'akaze').
            matcher: Matcher type ('bf', 'flann').
            ratio_test: Lowe's ratio threshold.

        Returns:
            MatchResult with keypoints, matches, and homography.
        """
        kp1, desc1 = self.extract(img1, method)
        kp2, desc2 = self.extract(img2, method)
        good = self.match_features(desc1, desc2, matcher, ratio_test, method)
        H = self.homography(kp1, kp2, good)

        return MatchResult(
            keypoints_src=kp1,
            keypoints_dst=kp2,
            matches=[],
            good_matches=good,
            homography_matrix=H,
        )

    def draw_matches(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        result: MatchResult,
        max_matches: int = 100,
    ) -> np.ndarray:
        """Draw feature matches between two images side by side.

        Args:
            img1: Source image.
            img2: Destination image.
            result: MatchResult from match_images().
            max_matches: Maximum matches to draw.

        Returns:
            Concatenated image with match lines.
        """
        good = result.good_matches[:max_matches]
        pts1 = np.float32([result.keypoints_src[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts2 = np.float32([result.keypoints_dst[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        canvas[:h1, :w1] = img1
        canvas[:h2, w1:w1 + w2] = img2

        for pt1, pt2 in zip(pts1, pts2):
            p1 = (int(pt1[0][0]), int(pt1[0][1]))
            p2 = (int(pt2[0][0]) + w1, int(pt2[0][1]))
            color = tuple(np.random.randint(0, 255, 3).tolist())
            cv2.circle(canvas, p1, 3, color, -1)
            cv2.circle(canvas, p2, 3, color, -1)
            cv2.line(canvas, p1, p2, color, 1)

        return canvas

    def panorama_stitch(
        self,
        images: list[np.ndarray],
        method: str = "sift",
        matcher: str = "flann",
    ) -> np.ndarray:
        """Stitch multiple images into a panorama.

        Sequentially warps images using homography. For more than 2 images,
        accumulates transforms from center outward.

        Args:
            images: List of images to stitch (ordered left to right).
            method: Feature detector.
            matcher: Feature matcher.

        Returns:
            Stitched panorama image.
        """
        if len(images) < 2:
            return images[0] if images else np.array([])

        result_img = images[0]
        for i in range(1, len(images)):
            match_result = self.match_images(result_img, images[i], method, matcher)

            if match_result.homography_matrix is None:
                print(f"Warning: Could not compute homography for image {i}. Skipping.")
                continue

            h2, w2 = images[i].shape[:2]
            h1, w1 = result_img.shape[:2]

            H = match_result.homography_matrix
            out_w = w1 + w2
            out_h = max(h1, h2)

            warped = cv2.warpPerspective(images[i], H, (out_w, out_h))
            warped[:h1, :w1] = result_img

            result_img = warped

        return result_img

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        """Convert to grayscale if needed."""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Feature detection and matching")
    parser.add_argument("images", nargs="+", help="Image paths (2+ for matching/stitching)")
    parser.add_argument("--method", default="sift", choices=["sift", "orb", "akaze"])
    parser.add_argument("--matcher", default="flann", choices=["bf", "flann"])
    parser.add_argument("--stitch", action="store_true", help="Stitch images into panorama")
    parser.add_argument("--output", default=None, help="Save output image")
    args = parser.parse_args()

    extractor = FeatureExtractor()

    loaded = []
    for p in args.images:
        img = cv2.imread(p)
        if img is None:
            print(f"Error: Cannot read {p}")
            return
        loaded.append(img)

    if args.stitch:
        panorama = extractor.panorama_stitch(loaded, args.method, args.matcher)
        if args.output:
            cv2.imwrite(args.output, panorama)
            print(f"Panorama saved to {args.output}")
        else:
            cv2.imshow("Panorama", panorama)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    elif len(loaded) >= 2:
        result = match_result = extractor.match_images(loaded[0], loaded[1], args.method, args.matcher)
        print(f"Keypoints: src={len(result.keypoints_src)}, dst={len(result.keypoints_dst)}")
        print(f"Good matches: {len(result.good_matches)}")
        print(f"Homography: {'Found' if result.homography_matrix is not None else 'Not found'}")

        vis = extractor.draw_matches(loaded[0], loaded[1], result)
        if args.output:
            cv2.imwrite(args.output, vis)
            print(f"Saved to {args.output}")
        else:
            cv2.imshow("Matches", vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        kp, desc = extractor.extract(loaded[0], args.method)
        print(f"Detected {len(kp)} keypoints with {args.method.upper()}")
        print(f"Descriptor shape: {desc.shape if desc is not None else 'None'}")


if __name__ == "__main__":
    main()
