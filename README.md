### 📊 Student Mental Health Analysis During Online Learning
**Thời gian**: 11/2025 | **Vai trò**: Project Leader
**Công cụ**: Python, MongoDB, Power BI, Machine Learning

**Mô tả dự án:**
Trong bối cảnh xã hội hiện nay, sức khỏe tinh thần ngày càng được xem là yếu tố quan trọng quyết định đến sự phát triển toàn diện sinh viên – những người đang trong giai đoạn định hướng nghề nghiệp và hình thành nhân cách. Áp lực học tập,khó khăn tài chính, cũng như ảnh hưởng của mạng xã hội đã khiến nhiều sinh viên rơi vào trạng thái căng thẳng.

## Các nhiệm vụ chính của dự án bao gồm:

**1. Data Cleaning**: Xử lý missing values, xóa outliers, chuẩn hóa dữ liệu bằng Python (Pandas, NumPy).

**2. Clustering Analysis**: Sử dụng K-Means để phân nhóm học sinh thành 5 groups dựa trên hành vi (Healthy, High-Risk, At-Risk, Sleep-Risk, Active).

**3. Dimensionality Reduction**: Áp dụng PCA để giảm từ 11 features → 3 components.

**4. Reporting**: Chuẩn bị Power BI Dashboard + PowerPoint presentation.

**5. Giới tính “other” và độ tuổi ảnh hưởng đến mức độ stress**: Phân tích dữ liệu dựa theo giới tính "other" từ đó xác định độ tuổi nào sẽ khả năng cao stress cao nhất khi học online.

**6. Hành vi (Sleep, Screen, Activity) ảnh hưởng tới mức độ stress**: Đánh giá yếu tố hành vi ảnh hưởng như thế nào đến tỷ lệ stress của học sinh/sinh viên.
## Insight tự dữ liệu:
<img width="2400" height="1948" alt="image" src="https://github.com/user-attachments/assets/73dcd7f8-590b-474b-834c-26880d3b5318" />
Kết Quả Chính

PC1 thể hiện yếu tố "Lối sống & Rủi ro sức khỏe": những loadings mạnh mẽ từ Screen_Sleep_Ratio (0.533), Health_Risk_Index (0.479) và Screen_Time (0.438) cho thấy mối liên hệ chặt chẽ giữa thời gian màn hình, hỏi thị tại giấc ngủ và rủi ro sức khỏe tổng thể.
PC2 đặc trưng cho "Tính cân bằng hoạt động - Tuổi tác": Age (0.424), Education_encoded (0.432) và Activity_Ratio (0.416) có loading dương mạnh, trong khi Sleep_Duration (-0.374) và Sleep_Quality_Score (-0.367) có loading âm, chỉ ra rằng những cá nhân cao tuổi hơn với trình độ giáo dục cao hơn có xu hướng có tỷ lệ hoạt động cao hơn nhưng chất lượng giấc ngủ thấp hơn.
PC3 tập trung vào "Cân bằng ngủ-tuổi tác": Age (0.565), Education_encoded (0.557) và Sleep_Duration (0.316) thể hiện mối quan hệ tích cực, cho thấy nhóm tuổi cao hơn có thể duy trì thời lượng giấc ngủ ổn định hơn.
Kết luận: Phân tích PCA thành công xác định ba chiều độc lập giải thích 73% phương sai, cung cấp cơ sở đáng tin cậy cho các can thiệp sức khỏe có mục tiêu.

## Mục tiêu dự án 
Nhận diện nhóm sinh viên có nguy cơ cao. Chỉ ra được ngoài những yếu tố hành vi còn những yêu tố nào có nguy cơ sức khỏe tinh thần học sinh/sinh viên khi học online từ đó tìm ra được những học sinh đang có vấn đề sức khỏe từ đó đưa ra những giải pháp can thiệp cụ thể.




