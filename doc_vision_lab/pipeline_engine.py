import cv2
import numpy as np

class PipelineEngine:
    """Manages and executes a sequence of OpenCV image processing operations."""

    AVAILABLE_STEPS = {
        "Grayscale": {},
        "CLAHE": {"clipLimit": 2.0, "tileGridSize": (8, 8)},
        "Gaussian Blur": {"ksize": (5, 5)},
        "Median Blur": {"ksize": 3},
        "Otsu Threshold": {},
        "Adaptive Threshold": {"blockSize": 11, "C": 2},
        "Morphological Dilation": {"kernel_size": (3, 3), "iterations": 1},
        "Morphological Erosion": {"kernel_size": (3, 3), "iterations": 1}
    }

    def __init__(self):
        self.steps = []

    def add_step(self, step_name: str, params: dict = None):
        """Adds a processing step to the pipeline."""
        if params is None:
            params = self.AVAILABLE_STEPS.get(step_name, {})
        self.steps.append({"step": step_name, "params": params})
        
    def load_pipeline(self, steps: list):
        """Loads a full pipeline list."""
        self.steps = steps

    def execute(self, image: np.ndarray) -> np.ndarray:
        """Executes all steps sequentially on the given image."""
        if image is None:
            return None
            
        current_img = image.copy()
        for step_dict in self.steps:
            name = step_dict["step"]
            p = step_dict["params"]
            current_img = self._apply_step(current_img, name, p)
        return current_img

    def _apply_step(self, img: np.ndarray, name: str, p: dict) -> np.ndarray:
        is_gray = len(img.shape) == 2

        if name == "Grayscale":
            if not is_gray: return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return img
        
        # Ensure grayscale for thresholding/CLAHE
        if name in ["CLAHE", "Otsu Threshold", "Adaptive Threshold"] and not is_gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            is_gray = True

        if name == "CLAHE":
            clahe = cv2.createCLAHE(clipLimit=p.get('clipLimit', 2.0), tileGridSize=tuple(p.get('tileGridSize', (8, 8))))
            return clahe.apply(img)
            
        elif name == "Gaussian Blur":
            return cv2.GaussianBlur(img, tuple(p.get('ksize', (5, 5))), 0)
            
        elif name == "Median Blur":
            return cv2.medianBlur(img, p.get('ksize', 3))
            
        elif name == "Otsu Threshold":
            _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return th
            
        elif name == "Adaptive Threshold":
            return cv2.adaptiveThreshold(
                img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, p.get('blockSize', 11), p.get('C', 2)
            )
            
        elif name == "Morphological Dilation":
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, tuple(p.get('kernel_size', (3, 3))))
            return cv2.dilate(img, kernel, iterations=p.get('iterations', 1))
            
        elif name == "Morphological Erosion":
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, tuple(p.get('kernel_size', (3, 3))))
            return cv2.erode(img, kernel, iterations=p.get('iterations', 1))
        
        return img
