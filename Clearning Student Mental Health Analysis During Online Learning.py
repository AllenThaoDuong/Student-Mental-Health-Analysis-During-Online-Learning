import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# load dữ liệu file CSV Steudent Mental Health Analysis During Online Learning vào trong python:
df = pd.read_csv(r'D:\Do An Cuoi Ky\Student Mental Health Analysis\Student Mental Health Analysis During Online Learning.csv')
# Hiện thị các thông tin cơ bản dữ liệu
print("Đọc dữ liệu và kiểm tra thông tin cơ bản về dữ liệu")
#1. Kiểm tra dữ liệu:
##1.2 Số lượng bản ghi và cột:
print(f'\nSố lượng bản ghi: {len(df)}')
print(f'Số lượng cột: {len(df.columns)}')
##1.3 Danh sách các cột:
print('\n Danh sách các cột:')
print(df.columns.tolist())
##1.4 Thông tin chi tiết vể dữ liệu:
print('\n Kiểu và thông tin dữ liệu:')
print(df.info())
##1.5 Thông tin 5 dòng đầu tiên:
print('\n 5 dòng đầu tiên của dữ liệu:')
print(df.head())
##1.6 Thống kê mô tả dữ liệu:
print('\nThống kê mô tả dữ liệu:')
print(df.describe(include='all'))

#1.7 Kiểm tra giá trị thiếu trong dữ liệu:
print('\nKiểm tra giá trị thiếu trong dữ liệu:')
missing_values = pd.DataFrame({
    "Cột": df.columns,
    "Số lượng giá trị thiếu": df.isnull().sum(),
    "Tỷ lệ phần trăm giá trị thiếu": (df.isnull().sum()/len(df))*100
})
print(missing_values)
#1.8 Kiểm tra dữ liệu trùng lặp:
print('\nKiểm tra dữ liệu trùng lặp:')
duplicate_count = df.duplicated().sum()
print(f'Số lượng bản ghi trùng lặp: {duplicate_count}')

#2. Kiểm tra giá trị bất thường:
print('\nKiểm tra giá trị bất thường:')

#2.1 Kiểm tra giá trị min và max của cột "Age":
print('\n1. Age - Kiểm tra giá trị bất thường:')
print(f'Min: {df["Age"].min()}, Max: {df["Age"].max()}')
print(f'Các giá trị Age > 30: {len(df[df["Age"] > 30])}')
print(f'Các giá trị Age >= 100: {len(df[df["Age"] >= 100])}')

#2.2 Kiểm tra giá trị bất thường của thời gian nhìn trên màn hình:
print('\n2. Screen Time - kiểm tra giá trị bất thường:')
print(f'Min: {df["Screen Time (hrs/day)"].min()}, Max: {df["Screen Time (hrs/day)"].max()}')
print(f'Các giá trị > 15 giờ/ngày: {len(df[df["Screen Time (hrs/day)"] > 15])}')

#2.3 Kiểm tra giá trị bất thường của thời gian ngủ:
print('\n3. Sleep Time - kiểm tra giá trị bất thường:')
print(f'Min: {df["Sleep Duration (hrs)"].min()}, Max: {df["Sleep Duration (hrs)"].max()}')
print(f'Các giá trị thời gian ngủ < 4 giờ/ngày: {len(df[df["Sleep Duration (hrs)"] < 4])}')
print(f'Các giá trị thời gian ngủ > 12 giờ/ngày: {len(df[df["Sleep Duration (hrs)"] > 12])}')

#2.4 Kiểm tra giá trị bất thường của thời gian hoạt đông thể chất:
print('\n4. Physical Activity Time - kiểm tra giá trị bất thường:')
print(f'Min: {df["Physical Activity (hrs/week)"].min()}, Max: {df["Physical Activity (hrs/week)"].max()}')
print(f'Các giá trị thời gian hoạt động thể chất > 50 giờ/Tuần: {len(df[df["Physical Activity (hrs/week)"] > 50])}')

#2.5 Phân phối các biến định lượng:
print('\nPhân phối các biến định lượng:')
categorical_cols = ['Gender','Education Level','Stress Level','Anxious Before Exams','Academic Performance Change']
for col in categorical_cols:
    print(f'\nPhân phối cua cột {col}:')
    print(df[col].value_counts().to_frame())
    print(f'Số lượng giá trị duy nhất: {df[col].nunique()}')

#3. Kiểm tra giá trị về chất lượng của data:
print('\nKiểm tra giá trị về chất lượng của data:')
#3.1 Kiểm tra Outliers của data:
print('\nKiểm tra Outliers của data:')
print('\Age Outliers (>=100):')
print(df[df['Age']>=100][["Name","Age",'Gender','Education Level']])

print('\nScreen Time Outliers (>15 hrs/day):')
print(df[df['Screen Time (hrs/day)']>15][["Name","Screen Time (hrs/day)",'Age','Gender','Education Level']])

print('\nSleep Duration Outliers (<4 or >12 hrs):')
print(df[(df['Sleep Duration (hrs)']<4) | (df['Sleep Duration (hrs)']>12)][["Name","Sleep Duration (hrs)",'Age','Gender','Education Level']])

print('\nPhysical Activity Outliers (>50 hrs/week):')
print(df[df['Physical Activity (hrs/week)']>50][["Name","Physical Activity (hrs/week)",'Age','Gender','Education Level']])

#3.2 Kiểm tra khoảng trắng thừa trong form dữ liệu text:
print('\nKiểm tra khoảng trắng thừa Education Level:')
edu_spaces = df[df['Education Level'].str.contains('^\s+', regex = True, na=False)]
if len(edu_spaces) > 0:
    print(df['Education Level'].unique()[:15])

#3.3 Xử lý giá trị không nhất quán trong định tính:
print('\nXử lý giá trị không nhất quán trong định tính:')
print('\nGender variations:')
print(df['Gender'].value_counts())

print('\nStress Level variations:')
print(df['Stress Level'].value_counts())

print('\nAnxious Before Exams variations:')
print(df['Anxious Before Exams'].value_counts())

print('\nAcademic Performance Change variations:')
print(df['Academic Performance Change'].value_counts())

#4. Giai đoạn tiếp theo sẽ là xử lý các vấn đề đã phát hiện trong quá trình kiểm tra dữ liệu: làm sạch dữ liệu, chuẩn hóa dữ liệu và xử lý các giá trị bất thường.
print("GIAI ĐOẠN: DATA CLEANING VỚI PYTHON")
#4.1 Xử lý missing data bằng median,mode hoặc mean:
print("\nBước 1: XỬ LÝ GIÁ TRỊ THIẾU")
from scipy import stats
# Tạo bản sao của dataframe gốc để làm sạch dữ liệu
df_cleaned = df.copy()
# Xử lý giá trị định lượng: dùng median hoặc mean
df_cleaned["Screen Time (hrs/day)"].fillna(df_cleaned["Screen Time (hrs/day)"].median(), inplace=True)
df_cleaned["Sleep Duration (hrs)"].fillna(df_cleaned["Sleep Duration (hrs)"].median(), inplace=True)
df_cleaned["Physical Activity (hrs/week)"].fillna(df_cleaned["Physical Activity (hrs/week)"].median(), inplace=True)

print('\nKiểm tra lại giá trị thiếu sau khi xử lý:')
print(df_cleaned.isnull().sum())
# xử lý dữ liệu trùng lặp:
df_cleaned.drop_duplicates(inplace=True)
print(f'Số lượng bản ghi sau khi loại bỏ trùng lặp: {len(df_cleaned)}')

print("\nBước 2: CHUẨN HÓA CATEGORICAL VALUES")
# 4.2 Chuẩn hóa dữ liệu định tính:
# 4.2.1 Chuẩn hóa Gender:
gender_mapping = {
    'Male':'Male','male':'Male','Mal':'Male',
    'Female':'Female','female':'Female','Femal':'Female',
    'Other':'Other'
}
df_cleaned['Gender'] = df_cleaned['Gender'].map(gender_mapping)
print('\nGender sau khi chuẩn hóa:')
print(df_cleaned['Gender'].unique())

# 4.2.2 Chuẩn hóa Stress Level:
stress_mapping = {
    'Low': 'Low', 'low': 'Low',
    'Medium': 'Medium', 'medium': 'Medium', 'MEDIUM': 'Medium',
    'High': 'High', 'high': 'High',
    'Very High': 'Very High',
    'Neutral': 'Medium'  # Neutral được gán thành Medium
}
df_cleaned['Stress Level'] = df_cleaned['Stress Level'].map(stress_mapping)
print('\nStress Level sau khi chuẩn hóa:')
print(df_cleaned['Stress Level'].unique())

# 4.2.3 Chuẩn hóa Anxious Before Exams:
#       Anxious Before Exams - standardize
anxious_mapping = {
    'Yes': 'Yes', 'YES': 'Yes', 'yes': 'Yes',
    'No': 'No', 'NO': 'No', 'no': 'No'
}
df_cleaned['Anxious Before Exams'] = df_cleaned['Anxious Before Exams'].map(anxious_mapping)
print(f"Anixious sau khi chuẩn hóa: {df_cleaned['Anxious Before Exams'].unique()}")

# 4.2.4 Chuẩn hóa Academic Performance Change:
#       Academic Performance Change
performance_mapping = {
    'Same': 'Same', 'same': 'Same',
    'Improved': 'Improved', 'Improvedd': 'Improved',
    'Declined': 'Declined', 'Declin': 'Declined'
}
df_cleaned['Academic Performance Change'] = df_cleaned['Academic Performance Change'].map(performance_mapping)
print(f"Academic Performance sau khi chuẩn hóa: {df_cleaned['Academic Performance Change'].unique()}")

# 4.2.5 Chuẩn hóa Education Level:
#       Education Level - loại bỏ khoảng trắng thừa
df_cleaned['Education Level'] = df_cleaned['Education Level'].str.strip()
print(f"Education Level sau khi chuẩn hóa: {df_cleaned['Education Level'].nunique()} unique values")

#5. Xử lý giá trị bất thường (outliers):
print("\nBước 3: XỬ LÝ OUTLIERS")

#5.1 Xử lý outliers của Age:
# Age: Phạm vị tuổi hợp lý từ 15 đến 35:
outlier_age = len(df_cleaned[(df_cleaned['Age'] < 15) | (df_cleaned['Age'] > 35)])
df_cleaned['Age'] = df_cleaned['Age'].clip(lower=15, upper=35)
print(f'Số lượng outliers Age đã được xử lý: {outlier_age}')

#5.2 Xử lý outliers của Screen Time:
# Screen time: Thời gian ngồi màn hình hợp lý từ 0 đến 15 giờ/ngày:
outlier_screen_time = len(df_cleaned[df_cleaned['Screen Time (hrs/day)'] > 15])
df_cleaned['Screen Time (hrs/day)'] = df_cleaned['Screen Time (hrs/day)'].clip(upper=15)
print(f'Số lượng outliers Screen Time đã được xử lý: {outlier_screen_time}')

#5.3 Xử lý outliers của Sleep Duration:
# Sleep Duration: Thời gian ngủ hợp lý từ 3 đến 12 giờ/ngày:
outlier_Sleep = len(df_cleaned[(df_cleaned['Sleep Duration (hrs)'] < 3) | (df_cleaned['Sleep Duration (hrs)'] > 12)])
df_cleaned['Sleep Duration (hrs)'] = df_cleaned['Sleep Duration (hrs)'].clip(lower=3, upper=12)
print(f'Số lượng outliers Sleep Duration đã được xử lý: {outlier_Sleep}')

#5.4 Xử lý outliers của Physical Activity:
# Physical Activity: Thời gian hoạt động thể chất hợp lý từ 0 đến 50 giờ/tuần:
outlier_Physical_Activity = len(df_cleaned[df_cleaned['Physical Activity (hrs/week)'] > 50])
df_cleaned['Physical Activity (hrs/week)'] = df_cleaned['Physical Activity (hrs/week)'].clip(upper=50)
print(f'Số lượng outliers Physical Activity đã được xử lý: {outlier_Physical_Activity}')

print("\nDữ liệu sau khi làm sạch:")
print(f'Ghi chép dữ liệu gốc: {len(df)}')
print(f'Ghi chép dữ liệu sau khi làm sạch: {len(df_cleaned)}')
print(f'Ghi chép dữ liệu thừa:{len(df) - len(df_cleaned)}')
print("\nThông tin dữ liệu sau khi làm sạch:")
print("\nDanh sách các cột sau khi làm sạch:")
print("\n5 dòng đầu tiên của dữ liệu sau khi làm sạch:")
print(df_cleaned.head())
print("\nThống kê mô tả dữ liệu sau khi làm sạch:")
print(df_cleaned.describe())
print("\n Đếm trong các cột Categori sau khi làm sạch:")
for col in categorical_cols:
    print(f'\nPhân phối cua cột {col}:')
    print(df_cleaned[col].value_counts())

# Tạo danh sách các biến định lượng vẽ biểu đồ phân phối
df_cleaned.columns = df_cleaned.columns.str.strip() # Đản bảo cột chính xác
num_cols = ["Screen Time (hrs/day)", "Sleep Duration (hrs)","Physical Activity (hrs/week)"]
# Khởi tạo biểu đồ phân phối cho các biến định lượng
plt.figure(figsize=(18, 5))
# Vẽ biểu đồ phân phối cho từng biến định lượng
for i, col in enumerate(num_cols):
    plt.subplot(1,3, i + 1)
    sns.boxplot(x='Stress Level', y=col, data=df_cleaned)
    plt.title(f'Phân phối của {col} theo mức độ căng thẳng')
    plt.xlabel('Stress Level')
    plt.ylabel(col)
plt.tight_layout()
plt.show()

# Bước 5: Xuất dữ liệu đã làm sạch ra file CSV mới
# Xuất dữ liệu đã làm sach ra file CSV:
df_cleaned.to_csv(r'D:\Do An Cuoi Ky\Student Mental Health Analysis\Student_Mental_Health_Analysis_Cleaned.csv', index=False)
print("\nDữ liệu đã làm sạch đã được xuất ra file CSV mới.")
# Xuất dữ liệu đã làm sach ra file JSON:
df_cleaned.to_json(r'D:\Do An Cuoi Ky\Student Mental Health Analysis\Student_Mental_Health_Analysis_Cleaned.json', orient='records', lines=True)
print("Dữ liệu đã làm sạch đã được xuất ra file JSON mới.")
