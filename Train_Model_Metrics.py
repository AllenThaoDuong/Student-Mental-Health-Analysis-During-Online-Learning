# 3 MODELS HOÀN CHỈNH
# K-Means + Random Forest + PCA
# ============================================================================

from os import name
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, confusion_matrix, classification_report, precision_recall_fscore_support
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("📖 BƯỚC CHUẨN BỊ: TẢI & XỬ LÝ DỮ LIỆU")
print("="*80)

# Tải dữ liệu
df = pd.read_csv('D:\Do An Cuoi Ky\Student Mental Health Analysis\Student_Mental_Health_Analysis_During_Online_Learning_Cleaned.csv')
print(f"\n✓ Tải dữ liệu: {df.shape[0]} học sinh, {df.shape[1]} cột")

# Mã hóa các cột categorical
gender_map = {'Male': 0, 'Female': 1, 'Other': 2}
df['Gender_encoded'] = df['Gender'].map(gender_map)

edu_order = {'Class 8': 1, 'Class 9': 2, 'Class 10': 3, 'Class 11': 4, 'Class 12': 5,
             'BA': 6, 'BSc': 6, 'BTech': 6, 'MA': 7, 'MSc': 7, 'MTech': 8}
df['Education_encoded'] = df['Education Level'].map(edu_order)

stress_map = {'Low': 0, 'Medium': 1, 'High': 2, 'Very High': 3}
df['Stress_encoded'] = df['Stress Level'].map(stress_map)

anxiety_map = {'No': 0, 'Yes': 1}
df['Anxiety_encoded'] = df['Anxious Before Exams'].map(anxiety_map)

print("✓ Mã hóa hoàn thành")

# Feature engineering
df['Sleep_Quality_Score'] = (df['Sleep Duration (hrs)'] / 6) * 100
df['Sleep_Quality_Score'] = df['Sleep_Quality_Score'].clip(0, 100)

df['Activity_Ratio'] = (df['Physical Activity (hrs/week)'] / 
                        (df['Screen Time (hrs/day)']*7 + 0.1))

df['Health_Risk_Index'] = ((df['Screen Time (hrs/day)'] - 
                            df['Sleep Duration (hrs)']) * 
                           (5 - df['Physical Activity (hrs/week)'] / 2))
df['Screen_Sleep_Ratio'] = df['Screen Time (hrs/day)'] / (df['Sleep Duration (hrs)'] + 0.1)

print("✓ Feature engineering hoàn thành")

# Chuẩn bị features
feature_cols = ['Age', 'Gender_encoded', 'Education_encoded', 
               'Screen Time (hrs/day)', 'Sleep Duration (hrs)', 
               'Physical Activity (hrs/week)', 'Anxiety_encoded',
               'Sleep_Quality_Score', 'Activity_Ratio', 'Health_Risk_Index',
               'Screen_Sleep_Ratio']

X = df[feature_cols]
y = df['Stress_encoded']

# Chuẩn hóa
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print(f"✓ Chuẩn hóa {len(feature_cols)} tính năng")

# PHẦN 1: K-MEANS CLUSTERING

print("\n" + "="*80)
print("🎯 PHẦN 1: K-MEANS CLUSTERING - PHÂN CỤM HỌC SINH")
print("="*80)

# Tìm K tối ưu
print("\n✓ Tìm số cluster tối ưu...")
X_weighted = X_scaled.copy()

screen_idx = feature_cols.index('Screen Time (hrs/day)')
sleep_idx = feature_cols.index('Sleep Duration (hrs)')
activity_idx = feature_cols.index('Activity_Ratio')
risk_idx = feature_cols.index('Health_Risk_Index')

X_weighted[:, screen_idx] *= 2.0
X_weighted[:, sleep_idx] *= 1.8
X_weighted[:, activity_idx] *= 1.8
X_weighted[:, risk_idx] *= 2.2

inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

optimal_k = K_range[np.argmax(silhouette_scores)]
print(f"✓ K tối ưu: {optimal_k} (Silhouette Score: {max(silhouette_scores):.4f})")
# Train K-Means
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_weighted)
df['Cluster'] = clusters

with open('kmeans_model.pkl', 'wb') as f:
    pickle.dump(kmeans, f)
print(f"✓ Train K-Means xong, lưu: kmeans_model.pkl")

# Đặt tên cho các cluster
print("\n✓ Phân tích & đặt tên cho các cluster...")
cluster_names = {
    0: {'name': "🟢 HEALTHY GROUP",   'desc': "Thói quen lành mạnh"},
    1: {'name': "🔴 HIGH-RISK GROUP", 'desc': "Screen time cao, ngủ ít"},
    2: {'name': "🟡 AT-RISK GROUP",   'desc': "Tiềm ẩn nguy cơ"},
    3: {'name': "💤 SLEEP-RISK GROUP",'desc': "Ngủ ít dù screen không cao"},
    4: {'name': "💪 ACTIVE GROUP",    'desc': "Vận động rất nhiều / outlier"},
}

for cluster_id in sorted(df['Cluster'].unique()):
    cluster_data = df[df['Cluster'] == cluster_id]
    n_students = len(cluster_data)
    
    avg_screen = cluster_data['Screen Time (hrs/day)'].mean()
    avg_sleep = cluster_data['Sleep Duration (hrs)'].mean()
    avg_activity = cluster_data['Physical Activity (hrs/week)'].mean()
    df['Cluster_Name'] = df['Cluster'].map(lambda x: cluster_names[x]['name'])
    print(f"\n  Cluster {cluster_id}: {name}")
    print(f"    Số học sinh: {n_students} ({n_students/len(df)*100:.1f}%)")
    print(f"    Màn hình: {avg_screen:.1f} hrs/day")
    print(f"    Ngủ: {avg_sleep:.1f} hrs/day")
    print(f"    Vận động: {avg_activity:.1f} hrs/week")

df['Cluster_Name'] = df['Cluster'].map(lambda x: cluster_names[x]['name'])

# Visualizations K-Means
print("\n✓ Vẽ biểu đồ K-Means...")

# 1. Elbow & Silhouette
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(K_range, inertias, 'bo-', linewidth=2.5, markersize=8)
ax1.axvline(x=optimal_k, color='r', linestyle='--', linewidth=2.5, label=f'Optimal K={optimal_k}')
ax1.set_xlabel('Số Cluster (K)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Inertia', fontsize=12, fontweight='bold')
ax1.set_title('📊 Elbow Method - Tìm điểm "khuỷu"', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=11)

ax2.plot(K_range, silhouette_scores, 'go-', linewidth=2.5, markersize=8)
ax2.axvline(x=optimal_k, color='r', linestyle='--', linewidth=2.5, label=f'Optimal K={optimal_k}')
ax2.set_xlabel('Số Cluster (K)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
ax2.set_title('📊 Silhouette Analysis - Chất lượng phân cụm', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=11)

plt.tight_layout()
plt.savefig('01_kmeans_elbow_silhouette.png', dpi=300, bbox_inches='tight')
print("  ✓ Lưu: 01_kmeans_elbow_silhouette.png")
plt.close()

# 2. Cluster Distribution
fig, ax = plt.subplots(figsize=(12, 6))

cluster_counts = df['Cluster'].value_counts().sort_index()
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'][:len(cluster_counts)]

labels_plot = [f"{cluster_names[i]['name']}\n({cluster_counts[i]/len(df)*100:.1f}%)" 
               for i in cluster_counts.index]

bars = ax.bar(range(len(cluster_counts)), cluster_counts.values,
              color=colors, edgecolor='black', linewidth=2, alpha=0.8)

ax.set_ylabel('Số học sinh', fontsize=12, fontweight='bold')
ax.set_title('📊 Phân bố học sinh theo Cluster', fontsize=14, fontweight='bold')
ax.set_xticks(range(len(cluster_counts)))
ax.set_xticklabels(labels_plot, fontsize=10)
ax.grid(axis='y', alpha=0.3)

for bar, count in zip(bars, cluster_counts.values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(count)}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('02_kmeans_cluster_distribution.png', dpi=300, bbox_inches='tight')
print("  ✓ Lưu: 02_kmeans_cluster_distribution.png")
plt.close()


# 3. Features comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

features_to_compare = [
    ('Screen Time (hrs/day)', 'Thời gian màn hình (hrs/day)'),
    ('Sleep Duration (hrs)', 'Giấc ngủ (hrs/day)'),
    ('Physical Activity (hrs/week)', 'Vận động (hrs/week)'),
    ('Health_Risk_Index', 'Chỉ số sức khỏe')
]

for idx, (feature, label) in enumerate(features_to_compare):
    ax = axes[idx // 2, idx % 2]
    
    means = [df[df['Cluster'] == c][feature].mean() for c in sorted(df['Cluster'].unique())]
    
    bars = ax.bar(range(len(means)), means, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
    
    ax.set_ylabel(label, fontsize=11, fontweight='bold')
    ax.set_title(f'{label} theo Cluster', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(means)))
    ax.set_xticklabels([f"C{i}" for i in sorted(df['Cluster'].unique())], fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.suptitle('📊 So sánh các tính năng chính theo Cluster', fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('04_kmeans_features_comparison.png', dpi=300, bbox_inches='tight')
print("  ✓ Lưu: 04_kmeans_features_comparison.png")
plt.close()

# 4. Stress distribution
fig, ax = plt.subplots(figsize=(12, 6))

stress_by_cluster = pd.crosstab(df['Cluster'], df['Stress Level'], normalize='index') * 100
stress_order = ['Low', 'Medium', 'High', 'Very High']
stress_by_cluster = stress_by_cluster[[col for col in stress_order if col in stress_by_cluster.columns]]

stress_colors_map = {'Low': '#2ECC71', 'Medium': '#F39C12', 'High': '#E74C3C', 'Very High': '#C0392B'}
stress_colors_list = [stress_colors_map[col] for col in stress_by_cluster.columns]

stress_by_cluster.plot(kind='bar', ax=ax, color=stress_colors_list, edgecolor='black', linewidth=1.5, width=0.8)

ax.set_xlabel('Cluster', fontsize=12, fontweight='bold')
ax.set_ylabel('Phần trăm (%)', fontsize=12, fontweight='bold')
ax.set_title('📊 Mức độ Stress theo Cluster', fontsize=14, fontweight='bold')
ax.legend(title='Stress Level', fontsize=10, title_fontsize=11)
ax.grid(axis='y', alpha=0.3)
ax.set_xticklabels([f"C{i}: {cluster_names[i]['name']}" for i in sorted(df['Cluster'].unique())],
                   rotation=45, ha='right')

plt.tight_layout()
plt.savefig('05_kmeans_stress_distribution.png', dpi=300, bbox_inches='tight')
print("  ✓ Lưu: 05_kmeans_stress_distribution.png")
plt.close()

# PHẦN 2: RANDOM FOREST

print("\n" + "="*80)
print("🎯 PHẦN 2: RANDOM FOREST - DỰ ĐOÁN MỨC ĐỘ STRESS")
print("="*80)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✓ Chia dữ liệu: Train={len(X_train)}, Test={len(X_test)}")

# Train RF
print("✓ Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
accuracy = (y_pred == y_test).sum() / len(y_test)

with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(rf, f)

print(f"✓ Độ chính xác: {accuracy:.2%}")
print(f"✓ Lưu: random_forest_model.pkl")

# Visualizations RF
print("\n✓ Vẽ biểu đồ Random Forest...")

# 1. Feature Importance
fig, ax = plt.subplots(figsize=(10, 8))

importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=True)

colors_imp = plt.cm.viridis(np.linspace(0, 1, len(importance_df)))

ax.barh(importance_df['Feature'], importance_df['Importance'], color=colors_imp, edgecolor='black', linewidth=1)
ax.set_xlabel('Độ quan trọng', fontsize=12, fontweight='bold')
ax.set_title('📊 Random Forest - Tính năng quan trọng nhất', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('06_rf_feature_importance.png', dpi=300, bbox_inches='tight')
print("  ✓ Lưu: 06_rf_feature_importance.png")
plt.close()

# 2. Confusion Matrix
fig, ax = plt.subplots(figsize=(10, 8))

stress_names = ['Low', 'Medium', 'High', 'Very High']
actual_classes = sorted(y_test.unique())
actual_names = [stress_names[i] for i in actual_classes]

cm = confusion_matrix(y_test, y_pred, labels=actual_classes)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=actual_names, yticklabels=actual_names,
            annot_kws={'size': 12, 'weight': 'bold'}, ax=ax)

ax.set_ylabel('Giá trị thực', fontsize=12, fontweight='bold')
ax.set_xlabel('Dự đoán', fontsize=12, fontweight='bold')
ax.set_title(f'📊 Confusion Matrix - Random Forest\nAccuracy: {accuracy:.2%}', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('07_rf_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("  ✓ Lưu: 07_rf_confusion_matrix.png")
plt.close()

# 3. Metrics by class
fig, ax = plt.subplots(figsize=(10, 6))

precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, labels=actual_classes)

metrics_df = pd.DataFrame({
    'Class': actual_names,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1
})

x = np.arange(len(metrics_df))
width = 0.25

ax.bar(x - width, metrics_df['Precision'], width, label='Precision', color='#FF6B6B', edgecolor='black')
ax.bar(x, metrics_df['Recall'], width, label='Recall', color='#FFA500', edgecolor='black')
ax.bar(x + width, metrics_df['F1-Score'], width, label='F1-Score', color='#4ECDC4', edgecolor='black')

ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('📊 Random Forest - Metrics theo từng lớp', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics_df['Class'])
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim([0, 1])

plt.tight_layout()
plt.savefig('08_rf_metrics_by_class.png', dpi=300, bbox_inches='tight')
print("  ✓ Lưu: 08_rf_metrics_by_class.png")
plt.close()

# PHẦN 3: PCA

print("\n" + "="*80)
print("🎯 PHẦN 3: PCA - GIẢM CHIỀU DỮ LIỆU")
print("="*80)

# Tìm optimal components
pca_full = PCA()
pca_full.fit(X_scaled)

variance_ratio = pca_full.explained_variance_ratio_
cumsum_variance = np.cumsum(variance_ratio)

n_components_85 = np.argmax(cumsum_variance >= 0.85) + 1

print(f"\n✓ Tìm components cần thiết cho 85% variance: {n_components_85}")

# Train PCA
pca = PCA(n_components=n_components_85)
X_pca = pca.fit_transform(X_scaled)

with open('pca_model.pkl', 'wb') as f:
    pickle.dump(pca, f)

print(f"✓ Train PCA xong")
print(f"  Chiều gốc: {X_scaled.shape[1]} → Chiều PCA: {X_pca.shape[1]}")
print(f"  Variance giữ lại: {pca.explained_variance_ratio_.sum()*100:.2f}%")
print(f"✓ Lưu: pca_model.pkl")

# Visualizations PCA
print("\n✓ Vẽ biểu đồ PCA...")

# 1. Variance explained
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

variance_to_plot = pca.explained_variance_ratio_
colors_pca = plt.cm.viridis(np.linspace(0, 1, n_components_85))

bars = ax1.bar(range(1, n_components_85 + 1), variance_to_plot * 100,
               color=colors_pca, edgecolor='black', linewidth=1.5)

ax1.set_xlabel('Principal Component', fontsize=12, fontweight='bold')
ax1.set_ylabel('Variance Explained (%)', fontsize=12, fontweight='bold')
ax1.set_title('📊 Variance từng Component', fontsize=13, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

for bar, var in zip(bars, variance_to_plot):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{var*100:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

cumsum_to_plot = np.cumsum(variance_to_plot)
ax2.plot(range(1, n_components_85 + 1), cumsum_to_plot * 100, 'bo-', linewidth=2.5, markersize=8)
ax2.axhline(y=85, color='r', linestyle='--', linewidth=2, label='85% Threshold')
ax2.fill_between(range(1, n_components_85 + 1), cumsum_to_plot * 100, alpha=0.3)

ax2.set_xlabel('Number of Components', fontsize=12, fontweight='bold')
ax2.set_ylabel('Cumulative Variance (%)', fontsize=12, fontweight='bold')
ax2.set_title('📊 Cumulative Variance', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 105])

plt.tight_layout()
plt.savefig('09_pca_variance_explained.png', dpi=300, bbox_inches='tight')
print("  ✓ Lưu: 09_pca_variance_explained.png")
plt.close()

# 2. Component Loadings
fig, ax = plt.subplots(figsize=(10, 8))

loadings_to_plot = pca.components_[:min(3, n_components_85)].T

sns.heatmap(loadings_to_plot, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
            xticklabels=[f'PC{i+1}' for i in range(min(3, n_components_85))],
            yticklabels=feature_cols, cbar_kws={'label': 'Loading'},
            annot_kws={'size': 9}, ax=ax, linewidths=0.5)

ax.set_title(f'📊 Component Loadings (Top {min(3, n_components_85)} PCs)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('10_pca_component_loadings.png', dpi=300, bbox_inches='tight')
print("  ✓ Lưu: 10_pca_component_loadings.png")
plt.close()

# 3. 2D Scatter
fig, ax = plt.subplots(figsize=(12, 8))

scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='RdYlGn_r',
                     s=80, alpha=0.6, edgecolors='black', linewidth=0.5)

ax.set_xlabel(f'PC1 ({variance_to_plot[0]*100:.1f}%)', fontsize=12, fontweight='bold')
ax.set_ylabel(f'PC2 ({variance_to_plot[1]*100:.1f}%)', fontsize=12, fontweight='bold')
ax.set_title('📊 PCA: 2D Projection (PC1 vs PC2)\nMàu = Stress Level', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Stress Level', fontsize=11, fontweight='bold')
cbar.set_ticks([0, 1, 2, 3])
cbar.set_ticklabels(['Low', 'Medium', 'High', 'Very High'])

plt.tight_layout()
plt.savefig('11_pca_2d_scatter.png', dpi=300, bbox_inches='tight')
print("  ✓ Lưu: 11_pca_2d_scatter.png")
plt.close()

# 4. 3D Scatter
if n_components_85 >= 3:
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
                         c=y, cmap='RdYlGn_r', s=80, alpha=0.6,
                         edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel(f'PC1 ({variance_to_plot[0]*100:.1f}%)', fontsize=11, fontweight='bold')
    ax.set_ylabel(f'PC2 ({variance_to_plot[1]*100:.1f}%)', fontsize=11, fontweight='bold')
    ax.set_zlabel(f'PC3 ({variance_to_plot[2]*100:.1f}%)', fontsize=11, fontweight='bold')
    ax.set_title('📊 PCA: 3D Projection (PC1, PC2, PC3)', fontsize=14, fontweight='bold', pad=20)
    
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5)
    cbar.set_label('Stress Level', fontsize=10, fontweight='bold')
    cbar.set_ticks([0, 1, 2, 3])
    cbar.set_ticklabels(['Low', 'Medium', 'High', 'Very High'])
    
    plt.tight_layout()
    plt.savefig('12_pca_3d_scatter.png', dpi=300, bbox_inches='tight')
    print("  ✓ Lưu: 12_pca_3d_scatter.png")
    plt.close()