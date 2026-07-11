class HypothesisGenerator:
    """Generates research hypotheses based on image quality metrics."""

    def __init__(self):
        # Thresholds can be tuned based on empirical testing
        self.BLUR_THRESHOLD = 150.0
        self.CONTRAST_THRESHOLD = 45.0
        self.NOISE_THRESHOLD = 8.0

    def generate(self, metrics: dict) -> list[str]:
        """Generate a list of hypotheses based on input metrics."""
        hypotheses = []
        blur = metrics.get("blur_score", 1000)
        contrast = metrics.get("contrast_score", 100)
        noise = metrics.get("noise_score", 0)

        if blur < self.BLUR_THRESHOLD:
            hypotheses.append(
                "Độ mờ (Blur) khá cao. "
                "**Giả thuyết:** Áp dụng Sharpening (Làm nét) hoặc Unsharp Mask có thể giúp khôi phục biên ký tự, làm giảm CER."
            )
        
        if contrast < self.CONTRAST_THRESHOLD:
            hypotheses.append(
                "Độ tương phản (Contrast) thấp. "
                "**Giả thuyết:** Áp dụng CLAHE hoặc Cân bằng Histogram (Histogram Equalization) sẽ giúp phân tách nền và chữ rõ ràng hơn trước khi OCR."
            )
            
        if noise > self.NOISE_THRESHOLD:
            hypotheses.append(
                "Mức độ nhiễu hạt (Noise) cao. "
                "**Giả thuyết:** Áp dụng Median Blur (Lọc trung vị) hoặc Gaussian Blur trước khi Thresholding sẽ giảm hiện tượng nét chữ bị vỡ vụn."
            )

        if not hypotheses:
            hypotheses.append(
                "Ảnh có chất lượng tương đối tốt. "
                "**Giả thuyết:** Có thể chạy trực tiếp OCR hoặc chỉ cần dùng Otsu Thresholding để tối ưu hóa việc phân tách nhị phân."
            )

        return hypotheses
