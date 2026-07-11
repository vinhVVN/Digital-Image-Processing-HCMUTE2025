from abc import ABC, abstractmethod
import numpy as np

class BaseOCRAdapter(ABC):
    """Abstract Base Class for OCR Adapters."""

    def __init__(self):
        self.model = None

    @abstractmethod
    def load_model(self):
        """Load the OCR model into memory. Should be implemented by subclasses."""
        pass

    @abstractmethod
    def extract_text(self, image: np.ndarray) -> tuple[str, float]:
        """
        Run OCR on the given image array.
        
        Args:
            image: numpy array (BGR or Grayscale)
            
        Returns:
            tuple: (extracted_text, average_confidence)
        """
        pass
