---
marp: true
theme: default
paginate: true
header: "Ứng dụng AI phát hiện bạo lực thời gian thực"
footer: "GVHD: TS. Ngô Dương Hà | SV: Nguyễn Minh Thọ - Đỗ Cao Thắng - Nguyễn Hoàng Tuấn"
---

<!-- _class: lead -->

# 1. ỨNG DỤNG AI PHÁT HIỆN BẠO LỰC THỜI GIAN THỰC TỪ VIDEO TRÊN NỀN TẢNG WEB VÀ MOBILE

**Giảng viên hướng dẫn:** TS. Ngô Dương Hà
**Sinh viên thực hiện:**

1. Nguyễn Minh Thọ (2001224971)
2. Đỗ Cao Thắng (2001224865)
3. Nguyễn Hoàng Tuấn (2001224555)

---

## 2. Lý do chọn đề tài & Thực trạng

- **Thực trạng cấp bách:**
  - Bạo lực nơi công cộng và học đường ngày càng gia tăng và diễn biến phức tạp.
  - Hệ thống camera giám sát (CCTV) bùng nổ về số lượng nhưng vẫn là "giám sát thụ động".
- **Vấn đề cốt lõi:**
  - Con người không thể theo dõi liên tục hàng trăm camera 24/7.
  - Phần lớn CCTV chỉ có tác dụng "xem lại" (pháp y) sau khi sự việc đã xảy ra.
- **Nhu cầu:**
  - Cần một giải pháp tự động hóa việc phát hiện.
  - Cảnh báo ngay lập tức (**Real-time**) để ngăn chặn rủi ro.

---

## 3. Mục tiêu và Phạm vi đề tài

### Mục tiêu chính

- Xây dựng hệ thống phát hiện bạo lực tự động với độ trễ thấp (< 1 giây).
- Tích hợp đa nền tảng: **Web Dashboard** (quản lý) và **Mobile App** (cảnh báo tức thời).
- Đảm bảo khả năng chịu tải cao và hoạt động ổn định.

### Phạm vi hệ thống

- Tập trung vào hành động bạo lực (Violence Detection).
- Kiểm chứng trên các tập dữ liệu chuẩn (**RWF-2000**, **Hockey**) và tập hợp nhất (**UVD**).

---

## 4. Công nghệ sử dụng

- **Trí tuệ nhân tạo (AI Core):**
  - PyTorch (Framework), **MobileNetV3** (Backbone), OpenCV (Xử lý ảnh).
- **Backend & Hệ thống:**
  - Python FastAPI (High performance API).
  - **Apache Kafka** (Message Queue - Xử lý luồng).
  - Redis (Caching & Pub/Sub).
  - Docker & Docker Compose (Containerization).
- **Cơ sở dữ liệu:**
  - Firebase Firestore (NoSQL - Realtime).
  - **HDFS** (Lưu trữ Big Data log).
- **Frontend:**
  - ReactJS + Vite (Web Admin).
  - Flutter (Cross-platform Mobile App).

---

## 5. Mô hình đề xuất - RemoNet (Cơ sở lý thuyết)

- **Cơ sở khoa học:** Dựa trên nghiên cứu của Huillcen et al. (MDPI Sensors 2024).
- **Tên bài báo:** "Efficient Human Violence Recognition for Surveillance in Real Time".
- **Kiến trúc áp dụng:** 3 khối xử lý tuần tự **SME -> STE -> GTE**.

_(Hình minh họa sơ đồ khối SME, STE, GTE)_

---

## 6. Chi tiết Module SME (Trích xuất chuyển động)

- **Nhiệm vụ:** Loại bỏ nhiễu nền (Background), chỉ giữ lại đối tượng chuyển động.
- **Thuật toán:**
  - Tính sai biệt giữa 2 frame liên tiếp (Optical Flow đơn giản).
  - Áp dụng ngưỡng (Thresholding) & Giãn nở (Dilation).
  - **Kết quả:** Loại bỏ các vùng tĩnh, chỉ giữ lại vùng chứa hành động quan trọng (giảm tải cho mô hình).

---

## 7. Chi tiết Module STE (Short-term Spatiotemporal)

- **STE (Short-term Spatiotemporal):**
  - Sử dụng Backbone **MobileNetV3-Small** (Pre-trained ImageNet).
  - Trích xuất đặc trưng hình ảnh của từng vùng chuyển động.

$$ p*t = \frac{1}{3} \sum*{c=1}^{3} (M^c_t) $$

_Trong đó: $M^c_t$ là giá trị pixel của kênh màu c tại thời điểm t._

---

## 8. Chi tiết Module GTE (Global Temporal - Spatial Pooling)

- **Nén không gian (Spatial Pooling):**

$$ S^c = \frac{1}{H \times W} \sum*{i=1}^{H} \sum*{j=1}^{W} B^c(i,j) $$

_Trong đó:_

- $S^c$ là giá trị đặc trưng trung bình của kênh c.
- $H, W$ là chiều cao và chiều rộng của bản đồ đặc trưng.

---

## 9. Chi tiết Module GTE (Temporal Compression)

- **Nén thời gian (Temporal Compression):** Tổng hợp thông tin từ chuỗi các frame để đưa ra quyết định cuối cùng.

$$ q = \frac{1}{C} \sum\_{c=1}^{C} S^c $$

-> Đầu ra cuối cùng là xác suất bạo lực (Violence Probability).

---

## 10. Đóng góp về dữ liệu - Bộ UVD

- **Vấn đề:** Các bộ dữ liệu cũ (Hockey, Movies) quá nhỏ hoặc dàn dựng. RWF-2000 thì thiếu đa dạng.
- **Giải pháp:** Xây dựng bộ **Unified Violence Dataset (UVD)**.
- **Thống kê:**
  - Tổng số: **5,000 videos**.
  - Nguồn: RWF-2000 + Hockey + RLVS (Real Life Violence).
  - Cân bằng dữ liệu: 50% Violence - 50% Non-violence.
  - -> Tăng khả năng tổng quát hóa (Generalization) của mô hình.

---

## 11. Kết quả thực nghiệm (So sánh Backbone)

**Bảng 1: So sánh trên tập Hockey Fight**
| Mô hình | Độ chính xác (%) | Params (Triệu) | FLOPs (G) |
| :--- | :---: | :---: | :---: |
| 3D-DenseNet | 97.0% | ~15.0 | ~45.0 |
| 3D-CNN End-to-end | 98.3% | ~35.0 | ~75.0 |
| **Đề xuất (MobileNetV3)** | **97.75%** | **2.54** | **1.25** |

**Bảng 2: So sánh trên tập RWF-2000**
| Mô hình | Độ chính xác (%) | Params (Triệu) | FLOPs (G) |
| :--- | :---: | :---: | :---: |
| I3D + RGB | 85.57 | 12.3 (Nặng) | 55.7 |
| SA+TA (SOTA) | 87.75 | 5.29 | 4.17 |
| **Đề xuất** | **82.00** | **2.54 (Nhẹ nhất)** | **1.25 (Nhanh nhất)** |

---

## 12. Kết quả thực nghiệm (Trên tập UVD)

**Bảng 3: Kết quả trên bộ dữ liệu hợp nhất (UVD - 5000 videos)**

| Chỉ số (Metric) | Giá trị đạt được | Nhận xét                 |
| :-------------- | :--------------: | :----------------------- |
| **Accuracy**    |    **87.78%**    | Tốt trên dữ liệu đa dạng |
| Recall          |      87.78%      | Bắt bạo lực rất nhạy     |
| Precision       |      87.87%      | Ít báo động giả          |

-> **Nhận xét:** Kết quả trên tập UVD đạt **87.78%**, thấp hơn Hockey (97.75%) do độ đa dạng cao hơn, nhưng Precision/Recall cân bằng (87.8%) cho thấy tính ổn định cao.

---

## 13. Phân tích định tính

- **Trường hợp đúng (True Positive):**
  - Nhận diện tốt các hành động nhanh: đấm, đá dứt khoát.
  - Ít bị nhiễu bởi thay đổi ánh sáng đột ngột.
- **Trường hợp sai (False Positive - Nhầm lẫn):**
  - Nhầm hành động "ôm nhau thắm thiết" thành "vật lộn".
  - Nhầm "nhảy múa đường phố" (có động tác vung tay mạnh) thành bạo lực.
- **Nguyên nhân:** Do mô hình dựa nhiều vào đặc trưng chuyển động (Motion), thiếu ngữ cảnh ngữ nghĩa sâu (Semantic).

---

## 14. Kiến trúc Pipeline thực tế (Tổng quan)

_(Hình sơ đồ khối tổng quát)_

- **Source:** Camera IP/RTSP.
- **Streaming:** RTSP Server -> Live Video Stream.
- **Processing:**
  - **Backend:** Đọc frame -> Phân phối qua Kafka.
  - **AI Service:** Tiêu thụ từ Kafka -> Buffer -> Xử lý (SME/STE/GTE).

---

## 15. Kiến trúc Pipeline thực tế (Chi tiết Streaming)

**Cập nhật quan trọng về luồng dữ liệu:**

1.  **RTSP Server:** Nhận luồng video gốc.
2.  **Backend Service:**
    - Đọc frame từ stream.
    - **Apache Kafka:** Hàng đợi bản tin (Message Queue). Frame được đẩy vào Topic `camera_frames`.
3.  **AI Consumer (Điều chỉnh so với slide cũ):**
    - **Buffer 30 frames:** _Nên được đặt tại đây (AI Service)_ thay vì Backend để đảm bảo tính toàn vẹn của cửa sổ trượt (Sliding Window).
    - Đọc từ Kafka -> Gom đủ 30 frame -> Gửi vào mô hình.

_(Lưu ý: Sơ đồ trong slide cũ đang để Buffer ở Backend, cần điều chỉnh lại cho đúng thực tế code)_

---

## 16. Sơ đồ Kiến trúc Hệ thống (Full System)

- **AI Detection Pipeline:**
  - Motion Detection (SME).
  - Feature Extraction (STE - MobileNetV3).
  - Classification (GTE).
- **Storage:**
  - **HDFS:** Lưu trữ toàn bộ kết quả (All Results) và log để huấn luyện lại.
  - **Redis:** Pub/Sub cho cảnh báo bạo lực (Violence Only).
- **Analytics Engine:**
  - Hotspot Clustering Model (Phân tích điểm nóng).
  - Risk Prediction Model (Dự báo rủi ro).
- **Applications:**
  - Mobile App (Alert, View Camera).
  - Admin Dashboard (View Camera, Alert/Report).

---

## 17. DEMO CHƯƠNG TRÌNH

- Trình diễn chức năng xem camera trực tiếp.
- Thử nghiệm kịch bản bạo lực giả lập.
- Nhận cảnh báo trên Mobile App và Web Dashboard.
- Xem lại biểu đồ phân tích (Analytics).

---

## 18. Kết luận & Hướng phát triển

- **Đóng góp chính:**
  - Đề xuất quy trình xử lý 3 bước (SME-STE-GTE) hiệu quả cho camera giám sát.
  - Xây dựng và chia sẻ bộ dữ liệu **UVD (5000 video)**.
  - Hệ thống minh họa hoàn chỉnh (Web/Mobile/BigData).
- **Hạn chế:** Chưa xử lý tốt video ban đêm (Night vision).
- **Hướng phát triển:**
  - Tối ưu hóa (Quantization) cho thiết bị biên (Edge AI).
  - Mở rộng nhận diện hành vi khác (té ngã, vũ khí).

---

## 19. LỜI CẢM ƠN

Chúng em xin chân thành cảm ơn:

- Giảng viên hướng dẫn **TS. Ngô Dương Hà** đã tận tình hướng dẫn, chỉ dạy cũng như chỉnh sửa những thứ còn thiếu sót trong quá trình chúng em thực hiện đề tài.
- Quý Thầy/Cô trong hội đồng đã lắng nghe và góp ý.

---

## 20. Q&A

**Hỏi đáp cùng hội đồng**
