import cv2
import numpy as np

class ImageAnalyzer:
    """Calculates image quality metrics relevant to OCR performance."""

    @staticmethod
    def compute_blur(image: np.ndarray) -> float:
        """
        Estimate blur using the Variance of Laplacian.
        Lower values indicate a blurrier image.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    @staticmethod
    def compute_contrast(image: np.ndarray) -> float:
        """
        Estimate contrast using the standard deviation of pixel intensities (RMS Contrast).
        Lower values indicate low contrast.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        return gray.std()

    @staticmethod
    def compute_noise(image: np.ndarray) -> float:
        """
        Estimate noise by calculating the standard deviation of the difference
        between the original image and a median blurred version.
        Higher values indicate more noise.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        blurred = cv2.medianBlur(gray, 3)
        diff = cv2.absdiff(gray, blurred)
        return diff.std()

    def analyze(self, image: np.ndarray) -> dict:
        """Returns a dictionary of quality metrics."""
        return {
            "blur_score": self.compute_blur(image),
            "contrast_score": self.compute_contrast(image),
            "noise_score": self.compute_noise(image)
        }
