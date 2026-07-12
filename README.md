# 🌟 Digital Image Processing HCMUTE 2025: Từ Thuật toán Cổ điển đến Hệ thống Deep Learning Vision

**Tác giả:** VinhVVN, HoangGJin

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=for-the-badge&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Data_Processing-013243?style=for-the-badge&logo=numpy&logoColor=white)
![PaddleOCR](https://img.shields.io/badge/PaddleOCR-Deep_Learning-red?style=for-the-badge&logo=paddlepaddle&logoColor=white)
![AI/ML](https://img.shields.io/badge/AI%2FML-Computer_Vision-FF6F00?style=for-the-badge)

## 📌 Tổng quan (Overview)

**Digital Image Processing HCMUTE 2025** là một dự án thị giác máy tính toàn diện, kết nối giữa các thuật toán xử lý ảnh cổ điển và các mô hình Deep Learning hiện đại. Được phát triển với trọng tâm là nền tảng thuật toán vững chắc, kiến trúc phần mềm có khả năng mở rộng và kỹ năng giải quyết vấn đề thực tế, kho lưu trữ này đóng vai trò vừa là môi trường nghiên cứu vừa là pipeline thị giác máy tính có thể triển khai thực tế.

Dự án khám phá một cách có hệ thống các phép biến đổi ảnh cốt lõi trước khi tiến tới các ứng dụng phức tạp trong thế giới thực như Phát hiện đối tượng (Object Detection) mạnh mẽ và hệ thống đánh giá Nhận dạng Ký tự Quang học (OCR) dạng module.

## ✨ Tính năng nổi bật (Key Features)

*   **Nền tảng Thuật toán Xử lý Ảnh**: Triển khai từ cơ bản đến tối ưu hóa bằng thư viện cho các phép biến đổi thiết yếu (Biến đổi điểm ảnh, Cân bằng Histogram) và lọc nâng cao (Không gian/Tần số, Butterworth, Inverse filter, Khử nhiễu Moire).
*   **Ứng dụng Computer Vision Thực tế**: Xây dựng các hệ thống toàn trình (end-to-end) cho Nhận diện hệ thống ngôn ngữ ký hiệu (ASL) theo thời gian thực, Phát hiện khuôn mặt (Face Detection) và Phát hiện trái cây (Fruit Detection) tự động.
*   **Hệ thống Nghiên cứu OCR Module (`Document Vision Lab`)**: Một kiến trúc pipeline đánh giá độc lập, có khả năng mở rộng cao được thiết kế chuyên biệt cho phân tích tài liệu.
*   **Design Patterns trong AI Engineering**: Triển khai **Adapter Pattern** để linh hoạt chuyển đổi giữa các mô hình OCR (ví dụ: PaddleOCR) mà không làm thay đổi logic đánh giá cốt lõi.
*   **Kiểm định Giả thuyết & Đánh giá Metrics**: Hệ thống đánh giá định lượng được tích hợp sẵn để đo lường hiệu suất của các mô hình vision dưới các điều kiện nhiễu và tiền xử lý ảnh khác nhau.

## 🏗️ Kiến trúc Dự án (Project Architecture)

Mã nguồn được tổ chức chặt chẽ nhằm tách biệt rõ ràng giữa logic xử lý cốt lõi, các tác vụ ứng dụng vision và các framework thử nghiệm.

```text
Digital-Image-Processing-HCMUTE2025/
│
├── core_processing/              # Chương 03, 04, 05
│   ├── point_transforms/         # Logarit, Lũy thừa, Phân ngưỡng
│   ├── histogram_ops/            # Thống kê, Cân bằng Histogram
│   ├── filtering/                # Không gian & Tần số (Gauss, Median, Highpass, Khử Moire)
│   └── morphology_features/      # Edge Detection, Sharpening
│
├── vision_applications/          # Ứng dụng CV trong thực tế
│   ├── asl_recognition/          # Nhận diện ngôn ngữ ký hiệu ASL
│   ├── face_detection/           # Theo dõi và nhận diện khuôn mặt
│   └── fruit_detection/          # Phát hiện đối tượng áp dụng cho trái cây
│
├── doc_vision_lab/               # Hệ thống Nghiên cứu & Đánh giá OCR (Headless)
│   ├── pipeline_engine.py        # Kiến trúc cốt lõi để đánh giá và xử lý
│   ├── quality_analyzer.py       # Đánh giá chất lượng & suy giảm ảnh
│   ├── metrics.py                # Hypothesis generator & đánh giá metrics
│   └── ocr_adapters/             # Tích hợp Design Patterns
│       ├── base.py               # Abstract Base Class cho OCR adapters
│       └── paddle_adapter.py     # Tích hợp PaddleOCR
│
├── MainHome.py                   # File chạy chính của ứng dụng
└── requirements.txt              # Các thư viện phụ thuộc của dự án
```

## 🛠️ Công nghệ sử dụng (Tech Stack)

*   **Ngôn ngữ:** Python 3.8+
*   **Computer Vision Cốt lõi:** OpenCV (cv2), NumPy, SciPy
*   **Deep Learning & OCR:** PaddleOCR
*   **Kiến trúc & Thiết kế:** Lập trình hướng đối tượng (OOP), Adapter Pattern, Tư duy thiết kế hướng miền (Domain-Driven Design)

## 🚀 Cài đặt & Sử dụng (Installation & Usage)

### 1. Thiết lập Môi trường

Khuyến nghị sử dụng môi trường ảo (virtual environment) để quản lý các thư viện một cách an toàn.

```bash
# Clone kho lưu trữ
git clone https://github.com/yourusername/Digital-Image-Processing-HCMUTE2025.git
cd Digital-Image-Processing-HCMUTE2025

# Tạo và kích hoạt môi trường ảo
python -m venv venv
# Trên Windows:
venv\Scripts\activate
# Trên Linux/macOS:
source venv/bin/activate

# Cài đặt các thư viện yêu cầu
pip install -r requirements.txt
```

### 2. Chạy Ứng dụng

Để khởi chạy giao diện ứng dụng chính và trải nghiệm các module xử lý:

```bash
python MainHome.py
```

## 📊 Kết quả & Demo (Results & Demo)

---
*Được xây dựng với niềm đam mê dành cho Computer Vision và AI Engineering.*
