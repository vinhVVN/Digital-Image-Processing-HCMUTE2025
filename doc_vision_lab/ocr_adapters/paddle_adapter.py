import numpy as np
from .base import BaseOCRAdapter

class PaddleAdapter(BaseOCRAdapter):
    """Adapter for PaddleOCR."""

    def load_model(self):
        if self.model is None:
            try:
                from paddleocr import PaddleOCR
                # use_angle_cls=True to handle rotated text, lang='vi' for Vietnamese
                self.model = PaddleOCR(use_angle_cls=True, lang='vi', enable_mkldnn=False)
            except ImportError:
                raise ImportError("Vui lòng cài đặt thư viện: pip install paddleocr paddlepaddle")

    def extract_text(self, image: np.ndarray) -> tuple[str, float]:
        if self.model is None:
            self.load_model()
            
        import cv2
        import numpy as np
        
        # 1. Normalize image type to uint8
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
            
        # 2. Force exactly 3 channels (BGR)
        if len(image.shape) == 2:
            img_input = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            img_input = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        elif len(image.shape) == 3 and image.shape[2] == 1:
            img_input = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            img_input = image.copy()
            
        # 3. Ensure C-contiguous array in memory (crucial for C++ backends like Paddle)
        img_input = np.ascontiguousarray(img_input)
            
        result = self.model.ocr(img_input)
        
        extracted_text = []
        confidences = []
        
        if not result:
            return "", 0.0

        # Handle different return formats from various PaddleOCR versions
        for res in result:
            if res is None:
                continue
                
            # Format 1: Dictionary (Newer PP-Structure/PaddleOCR versions)
            if isinstance(res, dict):
                texts = res.get('rec_texts', [])
                scores = res.get('rec_scores', [])
                for t, s in zip(texts, scores):
                    if t and str(t).strip():
                        extracted_text.append(str(t))
                        confidences.append(float(s))
                continue

            # Format 2: List of elements (Standard PaddleOCR)
            if isinstance(res, list):
                for line in res:
                    try:
                        if isinstance(line, (list, tuple)) and len(line) >= 2:
                            # Expected format: [box, (text, confidence)]
                            text_tuple = line[1]
                            if isinstance(text_tuple, (list, tuple)) and len(text_tuple) >= 2:
                                text = str(text_tuple[0])
                                conf = float(text_tuple[1])
                            else:
                                text = str(text_tuple)
                                conf = 1.0
                            
                            if text.strip():
                                extracted_text.append(text)
                                confidences.append(conf)
                    except Exception:
                        pass # Ignore parsing errors on weird lines
                
        final_text = " ".join(extracted_text)
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        return final_text, avg_conf
