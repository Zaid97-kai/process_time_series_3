import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import warnings
warnings.filterwarnings('ignore')
import os
import tempfile

# Загрузка данных
df = pd.read_excel('Dataset.xlsx')
print(f"Размер исходного датасета: {df.shape}")
print(f"Колонки: {df.columns.tolist()}")
print(f"\nУникальные файлы: {df['file_name'].nunique()}")
print(f"Статистика по количеству записей в файлах:")
file_stats = df['file_name'].value_counts()
print(file_stats.describe())

# Заменяем запятые на точки в числовых колонках (если нужно)
numeric_cols = ['left_ear', 'right_ear', 'mar']
for col in numeric_cols:
    if df[col].dtype == 'object':
        df[col] = df[col].str.replace(',', '.').astype(float)

# Входные признаки и целевые переменные - теперь все три параметра
feature_columns = ['left_ear', 'right_ear', 'mar']  # Все три параметра как входные
target_columns = ['left_ear', 'right_ear', 'mar']   # Все три параметра как целевые

# ================================================
# ФУНКЦИЯ СГЛАЖИВАНИЯ ДАННЫХ
# ================================================
def smooth_data(df, window_size=3, method='moving_average'):
    """
    Сглаживание данных для уменьшения шума и скачков
    """
    df_smoothed = df.copy()
    
    # Список колонок для сглаживания (все числовые колонки)
    columns_to_smooth = ['left_ear', 'right_ear', 'mar']
    
    # Применяем сглаживание отдельно для каждого файла
    for file_name in df_smoothed['file_name'].unique():
        file_mask = df_smoothed['file_name'] == file_name
        
        for col in columns_to_smooth:
            try:
                if method == 'moving_average':
                    # Простое скользящее среднее
                    df_smoothed.loc[file_mask, col] = df.loc[file_mask, col].rolling(
                        window=window_size, center=True, min_periods=1).mean()
                
                elif method == 'savitzky_golay':
                    # Фильтр Савицкого-Голея
                    try:
                        from scipy.signal import savgol_filter
                        window = min(window_size, len(df.loc[file_mask, col]))
                        if window % 2 == 0:
                            window -= 1  # Должно быть нечетным
                        window = max(3, window)
                        df_smoothed.loc[file_mask, col] = savgol_filter(
                            df.loc[file_mask, col], window, 2)
                    except ImportError:
                        print("Для метода savitzky_golay требуется scipy. Используется moving_average.")
                        df_smoothed.loc[file_mask, col] = df.loc[file_mask, col].rolling(
                            window=window_size, center=True, min_periods=1).mean()
                
                elif method == 'exponential':
                    # Экспоненциальное сглаживание
                    df_smoothed.loc[file_mask, col] = df.loc[file_mask, col].ewm(
                        span=window_size, adjust=False).mean()
            
            except Exception as e:
                print(f"Ошибка при сглаживании {col} в файле {file_name}: {e}")
                df_smoothed.loc[file_mask, col] = df.loc[file_mask, col]
    
    # Заполняем NaN значения (если остались)
    df_smoothed[columns_to_smooth] = df_smoothed[columns_to_smooth].fillna(method='ffill').fillna(method='bfill')
    
    return df_smoothed

# Визуализация сглаживания для случайных файлов
def visualize_smoothing(original_df, smoothed_df, file_names=None, save_path='smoothing_comparison.png'):
    """
    Визуализация сравнения оригинальных и сглаженных данных
    """
    if file_names is None:
        # Берем 4 случайных файла для визуализации
        unique_files = original_df['file_name'].unique()
        file_names = np.random.choice(unique_files, min(4, len(unique_files)), replace=False)
    
    n_files = len(file_names)
    
    fig, axes = plt.subplots(n_files, 3, figsize=(15, 4*n_files))
    
    if n_files == 1:
        axes = [axes]
    
    for file_idx, file_name in enumerate(file_names):
        orig_data = original_df[original_df['file_name'] == file_name]
        smooth_data = smoothed_df[smoothed_df['file_name'] == file_name]
        
        # График для left_ear
        axes[file_idx][0].plot(orig_data['frame_number'].values, orig_data['left_ear'].values, 
                              label='Original', alpha=0.7, linewidth=1)
        axes[file_idx][0].plot(smooth_data['frame_number'].values, smooth_data['left_ear'].values, 
                              label='Smoothed', alpha=0.9, linewidth=1.5)
        axes[file_idx][0].set_title(f'{file_name}\nleft_ear', fontsize=10)
        axes[file_idx][0].set_xlabel('Frame Number')
        axes[file_idx][0].set_ylabel('Value')
        axes[file_idx][0].legend(fontsize=8)
        axes[file_idx][0].grid(True, alpha=0.3)
        
        # График для right_ear
        axes[file_idx][1].plot(orig_data['frame_number'].values, orig_data['right_ear'].values, 
                              label='Original', alpha=0.7, linewidth=1)
        axes[file_idx][1].plot(smooth_data['frame_number'].values, smooth_data['right_ear'].values, 
                              label='Smoothed', alpha=0.9, linewidth=1.5)
        axes[file_idx][1].set_title(f'{file_name}\nright_ear', fontsize=10)
        axes[file_idx][1].set_xlabel('Frame Number')
        axes[file_idx][1].set_ylabel('Value')
        axes[file_idx][1].legend(fontsize=8)
        axes[file_idx][1].grid(True, alpha=0.3)
        
        # График для mar
        axes[file_idx][2].plot(orig_data['frame_number'].values, orig_data['mar'].values, 
                              label='Original', alpha=0.7, linewidth=1)
        axes[file_idx][2].plot(smooth_data['frame_number'].values, smooth_data['mar'].values, 
                              label='Smoothed', alpha=0.9, linewidth=1.5)
        axes[file_idx][2].set_title(f'{file_name}\nmar', fontsize=10)
        axes[file_idx][2].set_xlabel('Frame Number')
        axes[file_idx][2].set_ylabel('Value')
        axes[file_idx][2].legend(fontsize=8)
        axes[file_idx][2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"График сравнения сглаживания сохранен как '{save_path}'")

# Применяем сглаживание
print("\n--- Применение сглаживания данных ---")
df_smoothed = smooth_data(df, window_size=3, method='moving_average')

# Визуализируем результат сглаживания
visualize_smoothing(df, df_smoothed)

# ================================================
# РАЗДЕЛЕНИЕ ПО ФАЙЛАМ - КАЖДЫЙ ФАЙЛ = ОДИН СЕГМЕНТ
# ================================================

print("\n--- Разделение данных по файлам ---")
print("Каждый файл = один сегмент")

segments = []
file_info = []

for file_idx, file_name in enumerate(df_smoothed['file_name'].unique()):
    file_data = df_smoothed[df_smoothed['file_name'] == file_name].copy()
    
    # Сбрасываем индекс для каждого файла
    file_data = file_data.reset_index(drop=True)
    
    # Каждый файл - это отдельный сегмент
    segment_id = file_idx + 1
    file_data['segment_id'] = segment_id
    file_data['file_name_original'] = file_name
    
    segments.append(file_data)
    
    # Сохраняем информацию о файле/сегменте
    file_info.append({
        'segment_id': segment_id,
        'file_name': file_name,
        'frames_count': len(file_data),
        'min_frame': file_data['frame_number'].min(),
        'max_frame': file_data['frame_number'].max(),
        'avg_left_ear': file_data['left_ear'].mean(),
        'avg_right_ear': file_data['right_ear'].mean(),
        'avg_mar': file_data['mar'].mean()
    })
    
    print(f"  Сегмент {segment_id}: {file_name} - {len(file_data)} кадров")

print(f"\nСоздано {len(segments)} сегментов (файлов)")

# 2. Сохранение информации о сегментах
file_info_df = pd.DataFrame(file_info)
file_info_df.to_excel('segment_info.xlsx', index=False)
print("Информация о сегментах сохранена в 'segment_info.xlsx'")

# 3. Создание Excel файла с сегментами на разных листах
print("\nСохранение сегментов в Excel файл...")
with pd.ExcelWriter('segmented_time_series.xlsx') as writer:
    for i, segment in enumerate(segments):
        # Ограничиваем длину имени листа 31 символом (ограничение Excel)
        sheet_name = segment['file_name_original'].iloc[0]
        if len(sheet_name) > 31:
            sheet_name = sheet_name[:28] + "..."
        
        # Убираем недопустимые символы для имен листов
        invalid_chars = ['[', ']', ':', '*', '?', '/', '\\']
        for char in invalid_chars:
            sheet_name = sheet_name.replace(char, '_')
        
        segment.to_excel(writer, sheet_name=sheet_name, index=False)

print("Сегменты сохранены в 'segmented_time_series.xlsx'")

# ================================================
# ВЫЧИСЛЕНИЕ ВРЕМЕННЫХ ХАРАКТЕРИСТИК ДЛЯ КАЖДОГО ФАЙЛА/СЕГМЕНТА
# ================================================

def calculate_temporal_features(segment_df, feature_columns=['left_ear', 'right_ear', 'mar']):
    """
    Вычисляет временные характеристики для сегмента (файла)
    """
    features = {}
    
    # Базовые метаданные о сегменте
    features['frames_count'] = len(segment_df)
    features['duration_frames'] = segment_df['frame_number'].max() - segment_df['frame_number'].min() + 1
    
    for col in feature_columns:
        if col in segment_df.columns:
            data = segment_df[col].values
            
            # Базовые статистики для каждой колонки
            prefix = f"{col}_"
            features[prefix + 'min'] = np.min(data)
            features[prefix + 'max'] = np.max(data)
            features[prefix + 'mean'] = np.mean(data)
            features[prefix + 'median'] = np.median(data)
            features[prefix + 'std'] = np.std(data)
            features[prefix + 'variance'] = np.var(data)
            features[prefix + 'skewness'] = pd.Series(data).skew()
            features[prefix + 'kurtosis'] = pd.Series(data).kurtosis()
            features[prefix + 'autocorr'] = pd.Series(data).autocorr()
            features[prefix + 'area'] = np.trapz(data)
            
            # Линейный тренд
            if len(data) > 1:
                features[prefix + 'trend'] = np.polyfit(range(len(data)), data, 1)[0]
            else:
                features[prefix + 'trend'] = 0
            
            # Дополнительные характеристики
            features[prefix + 'range'] = np.max(data) - np.min(data)
            features[prefix + 'iqr'] = np.percentile(data, 75) - np.percentile(data, 25) if len(data) > 1 else 0
            features[prefix + 'mad'] = np.mean(np.abs(data - np.mean(data))) if len(data) > 0 else 0
            
            # Динамические характеристики (разности)
            if len(data) > 1:
                diff_data = np.diff(data)
                features[prefix + 'diff_mean'] = np.mean(diff_data)
                features[prefix + 'diff_std'] = np.std(diff_data)
                features[prefix + 'diff_max'] = np.max(np.abs(diff_data))
                features[prefix + 'diff_min'] = np.min(diff_data)
            else:
                features[prefix + 'diff_mean'] = 0
                features[prefix + 'diff_std'] = 0
                features[prefix + 'diff_max'] = 0
                features[prefix + 'diff_min'] = 0
            
            # Показатели стабильности
            if np.mean(data) != 0:
                features[prefix + 'cv'] = np.std(data) / np.mean(data)  # Коэффициент вариации
            else:
                features[prefix + 'cv'] = 0
    
    # Корреляция между признаками
    if len(feature_columns) >= 2 and all(col in segment_df.columns for col in feature_columns):
        if len(segment_df) > 1:
            corr_matrix = segment_df[feature_columns].corr()
            if len(feature_columns) >= 2:
                features['corr_left_right'] = corr_matrix.loc['left_ear', 'right_ear']
                features['corr_left_mar'] = corr_matrix.loc['left_ear', 'mar']
                features['corr_right_mar'] = corr_matrix.loc['right_ear', 'mar']
    
    return features

# Создание DataFrame с характеристиками для каждого сегмента (файла)
print("\n--- Вычисление временных характеристик для каждого файла ---")
features_list = []

for i, segment in enumerate(segments):
    features = calculate_temporal_features(segment, ['left_ear', 'right_ear', 'mar'])
    features['segment_id'] = i + 1
    features['file_name'] = segment['file_name_original'].iloc[0]
    features['total_frames'] = len(segment)
    features_list.append(features)
    
    if i < 5:  # Показываем информацию о первых 5 файлах
        print(f"  Сегмент {i+1}: {segment['file_name_original'].iloc[0]} - {len(segment)} кадров, "
              f"left_ear: {features['left_ear_mean']:.4f}, right_ear: {features['right_ear_mean']:.4f}, "
              f"mar: {features['mar_mean']:.4f}")

features_df = pd.DataFrame(features_list)

# Упорядочиваем колонки
id_cols = ['segment_id', 'file_name', 'total_frames', 'frames_count', 'duration_frames']
other_cols = [col for col in features_df.columns if col not in id_cols]
features_df = features_df[id_cols + other_cols]

# Сохранение характеристик в Excel
features_df.to_excel('temporal_features.xlsx', index=False)
print(f"\nВременные характеристики сохранены в 'temporal_features.xlsx'")
print(f"Размер таблицы характеристик: {features_df.shape}")
print(f"Количество признаков: {len(features_df.columns) - len(id_cols)}")

# ================================================
# АНАЛИЗ ДЛИН СЕГМЕНТОВ (ФАЙЛОВ)
# ================================================

print("\n--- Анализ длин сегментов ---")
segment_lengths = [len(segment) for segment in segments]
print(f"Минимальная длина: {min(segment_lengths)} кадров")
print(f"Максимальная длина: {max(segment_lengths)} кадров")
print(f"Средняя длина: {np.mean(segment_lengths):.1f} кадров")
print(f"Медианная длина: {np.median(segment_lengths)} кадров")
print(f"Стандартное отклонение: {np.std(segment_lengths):.1f} кадров")

# Визуализация распределения длин сегментов
plt.figure(figsize=(10, 6))
plt.hist(segment_lengths, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
plt.axvline(x=np.mean(segment_lengths), color='red', linestyle='--', 
           linewidth=2, label=f'Среднее: {np.mean(segment_lengths):.1f}')
plt.axvline(x=np.median(segment_lengths), color='green', linestyle='-', 
           linewidth=2, label=f'Медиана: {np.median(segment_lengths)}')
plt.xlabel('Длина сегмента (кадры)', fontsize=12)
plt.ylabel('Количество файлов', fontsize=12)
plt.title('Распределение длин сегментов (файлов)', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('segment_lengths_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("График распределения длин сохранен как 'segment_lengths_distribution.png'")

# ================================================
# КЛАСТЕРИЗАЦИЯ С ИСПОЛЬЗОВАНИЕМ DBSCAN
# ================================================

print("\n--- Кластеризация файлов по временным характеристикам с использованием DBSCAN ---")

# Подготовка данных для кластеризации (исключаем нечисловые и идентификационные колонки)
exclude_cols = ['segment_id', 'file_name', 'total_frames', 'frames_count', 'duration_frames']
numeric_features_df = features_df.select_dtypes(include=[np.number])
X = numeric_features_df.drop([col for col in exclude_cols if col in numeric_features_df.columns], 
                           axis=1, errors='ignore').values

if len(X) > 1 and X.shape[0] > 1:
    # Стандартизация данных
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"Размер данных для кластеризации: {X_scaled.shape}")
    print(f"Количество признаков для кластеризации: {X_scaled.shape[1]}")
    
    # Автоматический подбор параметров для DBSCAN
    print("Подбор параметров для DBSCAN...")
    
    # Пробуем разные значения eps
    eps_values = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    min_samples_values = [3, 5, 7, 10]
    
    best_eps = None
    best_min_samples = None
    best_n_clusters = 0
    best_labels = None
    
    for eps in eps_values:
        for min_samples in min_samples_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
            labels = dbscan.fit_predict(X_scaled)
            
            # Исключаем шумовые точки (метка -1)
            unique_labels = set(labels)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            
            # Оцениваем качество кластеризации
            if n_clusters > 1 and n_clusters <= min(10, len(X_scaled)//2):
                # Считаем процент шумовых точек
                noise_ratio = np.sum(labels == -1) / len(labels)
                
                if noise_ratio < 0.3:  # Не более 30% шума
                    if n_clusters > best_n_clusters:
                        best_n_clusters = n_clusters
                        best_eps = eps
                        best_min_samples = min_samples
                        best_labels = labels
    
    # Если не нашли хороших параметров, используем разумные значения по умолчанию
    if best_eps is None:
        best_eps = 1.0
        best_min_samples = 5
        print(f"Используем параметры по умолчанию: eps={best_eps}, min_samples={best_min_samples}")
        dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples, metric='euclidean')
        best_labels = dbscan.fit_predict(X_scaled)
        unique_labels = set(best_labels)
        best_n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    else:
        print(f"Найденные параметры: eps={best_eps}, min_samples={best_min_samples}")
        print(f"Количество кластеров: {best_n_clusters}")
    
    # Применяем DBSCAN с лучшими параметрами
    dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples, metric='euclidean')
    cluster_labels = dbscan.fit_predict(X_scaled)
    
    # Преобразуем метки: -1 становится отдельным кластером (шум)
    features_df['cluster'] = cluster_labels
    
    # Статистика по кластерам
    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
    n_noise = np.sum(cluster_labels == -1)
    
    print(f"\nРезультаты кластеризации DBSCAN:")
    print(f"  Количество кластеров (без учета шума): {n_clusters}")
    print(f"  Количество шумовых точек: {n_noise} ({n_noise/len(cluster_labels)*100:.1f}%)")
    print(f"  Параметры: eps={best_eps}, min_samples={best_min_samples}")
    
    # Переиндексируем кластеры, чтобы шум (-1) стал последним кластером
    cluster_mapping = {}
    new_cluster_labels = np.zeros_like(cluster_labels)
    
    # Присваиваем новые номера кластерам
    current_new_label = 0
    for label in unique_clusters:
        if label != -1:
            cluster_mapping[label] = current_new_label
            new_cluster_labels[cluster_labels == label] = current_new_label
            current_new_label += 1
    
    # Шумовые точки (-1) становятся отдельным кластером
    if -1 in unique_clusters:
        cluster_mapping[-1] = current_new_label
        new_cluster_labels[cluster_labels == -1] = current_new_label
    
    features_df['cluster'] = new_cluster_labels
    optimal_k = len(np.unique(new_cluster_labels))
    
    # Сохранение результатов кластеризации
    features_df.to_excel('temporal_features_with_clusters_dbscan.xlsx', index=False)
    
    print("\nРаспределение файлов по кластерам (после переиндексации):")
    cluster_distribution = features_df['cluster'].value_counts().sort_index()
    for cluster_id, count in cluster_distribution.items():
        percentage = count / len(features_df) * 100
        avg_frames = features_df[features_df['cluster'] == cluster_id]['total_frames'].mean()
        is_noise = " (шум)" if (cluster_id == optimal_k - 1 and -1 in unique_clusters) else ""
        print(f"  Кластер {cluster_id}{is_noise}: {count} файлов ({percentage:.1f}%), "
              f"средняя длина: {avg_frames:.1f} кадров")
    
    # Визуализация кластеров с помощью PCA (2D проекция)
    try:
        from sklearn.decomposition import PCA
        
        if X_scaled.shape[1] > 2:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            
            explained_variance = pca.explained_variance_ratio_
            print(f"PCA объясняет {explained_variance[0]*100:.1f}% и {explained_variance[1]*100:.1f}% дисперсии")
        else:
            X_pca = X_scaled
        
        plt.figure(figsize=(12, 8))
        
        # Создаем цветовую карту для кластеров
        unique_labels = np.unique(new_cluster_labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            if label == cluster_mapping.get(-1, -1):  # Шумовые точки
                color = 'gray'
                marker = 'x'
                size = 30
                alpha = 0.5
                label_text = f'Шум (кластер {label})'
            else:
                color = colors[i]
                marker = 'o'
                size = 50
                alpha = 0.7
                label_text = f'Кластер {label}'
            
            mask = new_cluster_labels == label
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                       c=[color], marker=marker, s=size, 
                       alpha=alpha, label=label_text, edgecolors='w', linewidth=0.5)
        
        plt.xlabel('Компонента 1', fontsize=12)
        plt.ylabel('Компонента 2', fontsize=12)
        plt.title(f'Визуализация кластеров DBSCAN ({optimal_k} кластеров)', fontsize=14)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        plt.savefig('cluster_visualization_dbscan.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Визуализация кластеров DBSCAN сохранена как 'cluster_visualization_dbscan.png'")
    except Exception as e:
        print(f"Не удалось создать визуализацию кластеров: {e}")
    
    # Анализ характеристик кластеров
    print("\nАнализ характеристик кластеров:")
    for cluster_id in range(optimal_k):
        cluster_data = features_df[features_df['cluster'] == cluster_id]
        is_noise = " (шум)" if (cluster_id == optimal_k - 1 and -1 in unique_clusters) else ""
        print(f"\nКластер {cluster_id}{is_noise} ({len(cluster_data)} файлов):")
        print(f"  Средняя длина: {cluster_data['total_frames'].mean():.1f} кадров")
        print(f"  Средний left_ear: {cluster_data['left_ear_mean'].mean():.4f}")
        print(f"  Средний right_ear: {cluster_data['right_ear_mean'].mean():.4f}")
        print(f"  Средний mar: {cluster_data['mar_mean'].mean():.4f}")
        
else:
    print("Недостаточно данных для кластеризации")
    features_df['cluster'] = 0
    optimal_k = 1
    dbscan = None
    scaler = None

# ================================================
# УЛУЧШЕННАЯ ПОДГОТОВКА ДАННЫХ ДЛЯ LSTM (ВСЕ 3 ПАРАМЕТРА)
# ================================================

def prepare_improved_lstm_data(segment_df, feature_columns, target_columns, sequence_length=10, 
                               validation_split=0.10, test_samples=30):
    """
    Улучшенная подготовка данных для обучения LSTM с нормализацией и разделением
    Теперь вход и выход - все три параметра
    """
    # Проверяем, что в сегменте достаточно данных
    min_required = sequence_length + 1 + test_samples
    if len(segment_df) < min_required:
        print(f"Предупреждение: сегмент содержит только {len(segment_df)} точек, "
              f"требуется минимум {min_required}")
        return None, None, None, None, None, None
    
    # Извлекаем только нужные колонки
    X_data = segment_df[feature_columns].values
    y_data = segment_df[target_columns].values
    
    X, y = [], []
    
    # Создаем последовательности
    for i in range(len(X_data) - sequence_length):
        X.append(X_data[i:i+sequence_length])  # Вход: sequence_length временных шагов
        y.append(y_data[i+sequence_length])    # Выход: следующий временной шаг
    
    if len(X) < 50:  # Минимальное количество последовательностей
        print(f"Слишком мало последовательностей: {len(X)}")
        return None, None, None, None, None, None
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Создано {len(X)} последовательностей")
    print(f"Размер входа: {X.shape}, размер выхода: {y.shape}")
    
    # Разделяем данные на train/validation/test
    total_samples = len(X)
    test_size = min(test_samples, total_samples // 5)  # Ограничиваем тестовую выборку
    val_size = int((total_samples - test_size) * validation_split)
    train_size = total_samples - test_size - val_size
    
    # Перемешиваем данные
    indices = np.arange(total_samples)
    np.random.shuffle(indices)
    
    X = X[indices]
    y = y[indices]
    
    # Разделение
    X_train, X_val, X_test = X[:train_size], X[train_size:train_size+val_size], X[train_size+val_size:train_size+val_size+test_size]
    y_train, y_val, y_test = y[:train_size], y[train_size:train_size+val_size], y[train_size+val_size:train_size+val_size+test_size]
    
    print(f"Разделение: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    
    # Нормализация данных (отдельно для каждого признака, на основе train)
    X_scalers = []
    y_scalers = []
    
    # Нормализация признаков (вход)
    X_train_scaled = np.zeros_like(X_train)
    X_val_scaled = np.zeros_like(X_val)
    X_test_scaled = np.zeros_like(X_test)
    
    for feature_idx in range(X_train.shape[2]):
        feature_scaler = MinMaxScaler()
        
        # Обучаем на тренировочных данных
        feature_data_train = X_train[:, :, feature_idx].reshape(-1, 1)
        feature_scaler.fit(feature_data_train)
        
        # Трансформируем все наборы данных
        X_train_scaled[:, :, feature_idx] = feature_scaler.transform(
            X_train[:, :, feature_idx].reshape(-1, 1)
        ).reshape(X_train.shape[0], X_train.shape[1])
        
        X_val_scaled[:, :, feature_idx] = feature_scaler.transform(
            X_val[:, :, feature_idx].reshape(-1, 1)
        ).reshape(X_val.shape[0], X_val.shape[1])
        
        X_test_scaled[:, :, feature_idx] = feature_scaler.transform(
            X_test[:, :, feature_idx].reshape(-1, 1)
        ).reshape(X_test.shape[0], X_test.shape[1])
        
        X_scalers.append(feature_scaler)
    
    # Нормализация целевых переменных (выход)
    y_train_scaled = np.zeros_like(y_train)
    y_val_scaled = np.zeros_like(y_val)
    y_test_scaled = np.zeros_like(y_test)
    
    for target_idx in range(y_train.shape[1]):
        target_scaler = MinMaxScaler()
        
        # Обучаем на тренировочных данных
        target_data_train = y_train[:, target_idx].reshape(-1, 1)
        target_scaler.fit(target_data_train)
        
        # Трансформируем все наборы данных
        y_train_scaled[:, target_idx] = target_scaler.transform(
            y_train[:, target_idx].reshape(-1, 1)
        ).flatten()
        
        y_val_scaled[:, target_idx] = target_scaler.transform(
            y_val[:, target_idx].reshape(-1, 1)
        ).flatten()
        
        y_test_scaled[:, target_idx] = target_scaler.transform(
            y_test[:, target_idx].reshape(-1, 1)
        ).flatten()
        
        y_scalers.append(target_scaler)
    
    return (X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, 
            X_test_scaled, y_test_scaled, X_scalers, y_scalers)

# ================================================
# УЛУЧШЕННАЯ АРХИТЕКТУРА LSTM С РЕГУЛЯРИЗАЦИЕЙ (3 ВЫХОДА)
# ================================================

def create_improved_lstm_model(input_shape, output_shape, dropout_rate=0.3, 
                              recurrent_dropout=0.2, l2_reg=0.001):
    """
    Создает улучшенную модель LSTM с регуляризацией для прогнозирования 3 параметров
    """
    from tensorflow.keras.regularizers import l2
    
    model = Sequential([
        # Первый LSTM слой с регуляризацией
        LSTM(128, activation='tanh', return_sequences=True,
             input_shape=input_shape,
             kernel_regularizer=l2(l2_reg),
             recurrent_regularizer=l2(l2_reg),
             dropout=dropout_rate,
             recurrent_dropout=recurrent_dropout),
        
        Dropout(dropout_rate),
        
        # Второй LSTM слой
        LSTM(64, activation='tanh', return_sequences=False,
             kernel_regularizer=l2(l2_reg),
             recurrent_regularizer=l2(l2_reg),
             dropout=dropout_rate,
             recurrent_dropout=recurrent_dropout),
        
        Dropout(dropout_rate),
        
        # Полносвязные слои
        Dense(64, activation='relu', kernel_regularizer=l2(l2_reg)),
        Dropout(dropout_rate),
        
        Dense(32, activation='relu', kernel_regularizer=l2(l2_reg)),
        Dropout(dropout_rate),
        
        # Выходной слой с 3 нейронами (left_ear, right_ear, mar)
        Dense(output_shape)
    ])
    
    return model

# ================================================
# КОМПЛЕКСНАЯ СТРАТЕГИЯ ОБУЧЕНИЯ (ДЛЯ 3 ВЫХОДОВ)
# ================================================

def train_lstm_with_strategy(X_train, y_train, X_val, y_val, 
                            feature_columns, target_columns,
                            cluster_id, file_name):
    """
    Обучает LSTM с различными стратегиями и выбирает лучшую модель
    """
    import tempfile
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    output_shape = len(target_columns)  # Теперь 3 выхода
    
    strategies = [
        {'name': 'baseline', 'dropout': 0.2, 'l2_reg': 0.0, 'lr': 0.001},
        {'name': 'regularized', 'dropout': 0.3, 'l2_reg': 0.001, 'lr': 0.001},
        {'name': 'high_reg', 'dropout': 0.4, 'l2_reg': 0.01, 'lr': 0.0005},
    ]
    
    best_model = None
    best_val_loss = float('inf')
    best_strategy_name = ''
    history_results = {}
    
    for strategy in strategies:
        print(f"\n  Тестируем стратегию: {strategy['name']}")
        print(f"    Dropout: {strategy['dropout']}, L2: {strategy['l2_reg']}, LR: {strategy['lr']}")
        
        # Создаем модель
        model = create_improved_lstm_model(
            input_shape, output_shape,
            dropout_rate=strategy['dropout'],
            l2_reg=strategy['l2_reg']
        )
        
        # Компилируем с текущей стратегией
        model.compile(
            optimizer=Adam(learning_rate=strategy['lr']),
            loss='mse',  # MSE для многомерного выхода
            metrics=['mae']  # MAE для каждого выхода
        )
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=0
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=0
        )
        
        # Создаем временный файл для сохранения лучшей модели
        temp_dir = tempfile.mkdtemp()
        checkpoint_path = os.path.join(temp_dir, f'best_model_{strategy["name"]}.keras')
        
        model_checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=0
        )
        
        # Обучение
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr, model_checkpoint],
            verbose=0
        )
        
        # Оцениваем модель
        val_loss = history.history['val_loss'][-1]
        history_results[strategy['name']] = {
            'val_loss': val_loss,
            'train_loss': history.history['loss'][-1],
            'epochs': len(history.history['loss'])
        }
        
        print(f"    Результат: val_loss={val_loss:.6f}, epochs={len(history.history['loss'])}")
        
        # Сохраняем лучшую модель
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
            best_strategy_name = strategy['name']
            
            # Загружаем лучшие веса
            if os.path.exists(checkpoint_path):
                best_model.load_weights(checkpoint_path)
        
        # Удаляем временный файл
        try:
            os.remove(checkpoint_path)
            os.rmdir(temp_dir)
        except:
            pass
    
    print(f"\n  Лучшая стратегия: {best_strategy_name} с val_loss={best_val_loss:.6f}")
    
    return best_model, history_results

# ================================================
# АНАЛИЗ И ПРЕДОБРАБОТКА ДАННЫХ ПЕРЕД ОБУЧЕНИЕМ
# ================================================

def analyze_and_preprocess_segment(segment, feature_columns, target_columns):
    """
    Анализирует сегмент и выполняет предобработку перед обучением
    """
    print(f"\n  Анализ сегмента перед обучением:")
    
    # Анализ данных
    stats = segment[feature_columns].describe()
    print(f"    Статистика признаков:")
    for col in feature_columns:
        print(f"      {col}: mean={segment[col].mean():.4f}, std={segment[col].std():.4f}, "
              f"min={segment[col].min():.4f}, max={segment[col].max():.4f}")
    
    # Проверка на выбросы
    segment_copy = segment.copy()
    for col in feature_columns:
        q1 = segment_copy[col].quantile(0.25)
        q3 = segment_copy[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = segment_copy[(segment_copy[col] < lower_bound) | (segment_copy[col] > upper_bound)]
        if len(outliers) > 0:
            print(f"    ВНИМАНИЕ: {len(outliers)} выбросов в {col} "
                  f"({len(outliers)/len(segment_copy)*100:.1f}%)")
            
            # Обработка выбросов - winsorization
            segment_copy.loc[segment_copy[col] < lower_bound, col] = lower_bound
            segment_copy.loc[segment_copy[col] > upper_bound, col] = upper_bound
    
    return segment_copy

# ================================================
# ВИЗУАЛИЗАЦИЯ ПРОГНОЗОВ ДЛЯ ВСЕГО СЕГМЕНТА (3 ПАРАМЕТРА)
# ================================================

def visualize_segment_predictions(segment_df, predictions, actuals, cluster_id, file_name, 
                                 feature_names, sequence_length=10):
    """
    Создает график с прогнозами для всего сегмента для всех 3 параметров
    """
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # Получаем временные метки
    time_indices = np.arange(len(predictions))
    
    # Для каждого параметра создаем свой график
    for idx, param_name in enumerate(feature_names):
        actual_values = actuals[:, idx]
        predicted_values = predictions[:, idx]
        
        # График прогнозов vs фактические значения
        axes[idx].plot(time_indices, actual_values, label='Фактические значения', 
                      color='blue', alpha=0.7, linewidth=2)
        axes[idx].plot(time_indices, predicted_values, label='Прогнозы LSTM', 
                      color='red', alpha=0.7, linewidth=2, linestyle='--')
        
        # Заполняем область между линиями
        axes[idx].fill_between(time_indices, actual_values, predicted_values, 
                             where=(predicted_values >= actual_values), 
                             color='red', alpha=0.2, label='Переоценка')
        axes[idx].fill_between(time_indices, actual_values, predicted_values, 
                             where=(predicted_values < actual_values), 
                             color='blue', alpha=0.2, label='Недооценка')
        
        # Расчет ошибок
        errors = predicted_values - actual_values
        mape = np.mean(np.abs(errors / (actual_values + 1e-10))) * 100
        mae = np.mean(np.abs(errors))
        
        # Добавляем информацию о качестве
        axes[idx].text(0.02, 0.95, f'{param_name}\nMAPE: {mape:.2f}%\nMAE: {mae:.4f}', 
                     transform=axes[idx].transAxes,
                     fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        axes[idx].set_xlabel('Временной индекс', fontsize=10)
        axes[idx].set_ylabel(param_name, fontsize=10)
        axes[idx].legend(loc='best')
        axes[idx].grid(True, alpha=0.3)
    
    plt.suptitle(f'Прогнозы LSTM для всего сегмента (Кластер {cluster_id})\n{file_name[:50]}...', 
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'segment_predictions_cluster_{cluster_id}_all_params.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Создаем суммарную метрику
    total_mape = np.mean([np.mean(np.abs(predictions[:, i] - actuals[:, i]) / (actuals[:, i] + 1e-10)) * 100 
                          for i in range(len(feature_names))])
    total_mae = np.mean([np.mean(np.abs(predictions[:, i] - actuals[:, i])) 
                         for i in range(len(feature_names))])
    
    return total_mape, total_mae

# ================================================
# ПРОГНОЗИРОВАНИЕ ДЛЯ ОДНОГО СЕГМЕНТА КАЖДОГО КЛАСТЕРА (3 ПАРАМЕТРА)
# ================================================

def predict_for_single_segment(cluster_id, segment_df, model, X_scalers, y_scalers, 
                              feature_columns, target_columns, sequence_length=10):
    """
    Прогнозирует значения для одного сегмента с использованием обученной модели
    Возвращает прогнозы для всех 3 параметров
    """
    # Подготавливаем данные для прогнозирования
    X_data = segment_df[feature_columns].values
    y_data = segment_df[target_columns].values
    
    predictions = []
    actuals = []
    
    # Прогнозируем для всех возможных последовательностей
    for i in range(sequence_length, len(X_data)):
        # Берем последовательность
        X_sequence = X_data[i-sequence_length:i]
        
        # Масштабируем
        X_scaled = np.zeros((1, sequence_length, len(feature_columns)))
        for feature_idx in range(len(feature_columns)):
            feature_scaler = X_scalers[feature_idx]
            feature_data = X_sequence[:, feature_idx].reshape(-1, 1)
            feature_scaled = feature_scaler.transform(feature_data).flatten()
            X_scaled[0, :, feature_idx] = feature_scaled
        
        # Прогнозируем
        y_pred_scaled = model.predict(X_scaled, verbose=0)
        
        # Обратное преобразование масштаба
        y_pred = np.zeros(len(target_columns))
        for target_idx in range(len(target_columns)):
            y_pred[target_idx] = y_scalers[target_idx].inverse_transform(
                y_pred_scaled[:, target_idx].reshape(-1, 1)
            ).flatten()[0]
        
        predictions.append(y_pred)
        actuals.append(y_data[i])
    
    # Создаем DataFrame с результатами
    predictions_array = np.array(predictions)
    actuals_array = np.array(actuals)
    
    results_dict = {
        'frame_number': segment_df['frame_number'].iloc[sequence_length:].values
    }
    
    # Добавляем данные для каждого параметра
    for idx, param_name in enumerate(target_columns):
        results_dict[f'actual_{param_name}'] = actuals_array[:, idx]
        results_dict[f'predicted_{param_name}'] = predictions_array[:, idx]
        results_dict[f'error_{param_name}'] = predictions_array[:, idx] - actuals_array[:, idx]
        results_dict[f'absolute_error_{param_name}'] = np.abs(predictions_array[:, idx] - actuals_array[:, idx])
    
    results_df = pd.DataFrame(results_dict)
    
    # Расчет метрик для каждого параметра
    metrics = {}
    for idx, param_name in enumerate(target_columns):
        actual_values = actuals_array[:, idx]
        predicted_values = predictions_array[:, idx]
        
        if len(actual_values) > 0:
            errors = predicted_values - actual_values
            
            # Защита от деления на ноль
            valid_idx = np.abs(actual_values) > 1e-10
            if np.any(valid_idx):
                mape = np.mean(np.abs(errors[valid_idx] / actual_values[valid_idx])) * 100
            else:
                mape = np.mean(np.abs(errors)) * 100
            
            mae = np.mean(np.abs(errors))
            rmse = np.sqrt(np.mean(errors**2))
            
            metrics[param_name] = {
                'mape': mape,
                'mae': mae,
                'rmse': rmse
            }
    
    # Общие метрики
    metrics['total'] = {
        'total_predictions': len(predictions),
        'avg_mape': np.mean([metrics[param]['mape'] for param in target_columns]),
        'avg_mae': np.mean([metrics[param]['mae'] for param in target_columns]),
        'avg_rmse': np.mean([metrics[param]['rmse'] for param in target_columns])
    }
    
    return results_df, metrics, predictions_array, actuals_array

# ================================================
# ОБУЧЕНИЕ LSTM С УЛУЧШЕННОЙ СТРАТЕГИЕЙ (3 ВЫХОДА)
# ================================================

print("\n" + "="*60)
print("УЛУЧШЕННОЕ ОБУЧЕНИЕ LSTM МОДЕЛЕЙ (3 ВЫХОДА)")
print("="*60)

lstm_models = {}
lstm_scalers_X = {}
lstm_scalers_y = {}
lstm_mape_results = {}
cluster_segment_samples = {}
training_strategies = {}  # Сохраняем использованные стратегии
segment_predictions = {}  # Сохраняем прогнозы для каждого кластера

for cluster_id in range(optimal_k):
    print(f"\n--- Обучение LSTM для кластера {cluster_id} ---")
    
    # Получаем сегменты (файлы) текущего кластера
    cluster_segment_ids = features_df[features_df['cluster'] == cluster_id]['segment_id'].values
    cluster_segment_ids = [int(id) - 1 for id in cluster_segment_ids]
    
    if len(cluster_segment_ids) == 0:
        print(f"Кластер {cluster_id} пустой, пропускаем")
        continue
    
    # Анализируем все файлы в кластере для выбора лучшего
    print(f"  Анализ файлов в кластере {cluster_id} ({len(cluster_segment_ids)} файлов):")
    
    segment_stats = []
    for seg_id in cluster_segment_ids:
        if seg_id < len(segments):
            segment = segments[seg_id]
            stats = {
                'segment_id': seg_id + 1,
                'file_name': segment['file_name_original'].iloc[0],
                'frames': len(segment),
                'left_ear_mean': segment['left_ear'].mean(),
                'right_ear_mean': segment['right_ear'].mean(),
                'mar_mean': segment['mar'].mean(),
                'left_ear_std': segment['left_ear'].std(),
                'right_ear_std': segment['right_ear'].std(),
                'mar_std': segment['mar'].std(),
                'sequences': max(0, len(segment) - 10)  # Оценка количества последовательностей
            }
            segment_stats.append(stats)
    
    # Выбираем лучший файл для обучения (не случайный, а наиболее репрезентативный)
    segment_stats_df = pd.DataFrame(segment_stats)
    
    # Критерии выбора:
    # 1. Достаточное количество данных
    # 2. Умеренная дисперсия (не слишком шумный)
    # 3. Средние значения близки к средним по кластеру
    cluster_left_mean = features_df[features_df['cluster'] == cluster_id]['left_ear_mean'].mean()
    cluster_right_mean = features_df[features_df['cluster'] == cluster_id]['right_ear_mean'].mean()
    cluster_mar_mean = features_df[features_df['cluster'] == cluster_id]['mar_mean'].mean()
    
    # Вычисляем рейтинг для каждого сегмента
    segment_stats_df['data_sufficiency'] = segment_stats_df['sequences'] / segment_stats_df['sequences'].max()
    segment_stats_df['stability'] = 1 / (1 + (segment_stats_df['left_ear_std'] + 
                                            segment_stats_df['right_ear_std'] + 
                                            segment_stats_df['mar_std']) / 3)
    segment_stats_df['representativeness'] = 1 / (1 + (abs(segment_stats_df['left_ear_mean'] - cluster_left_mean) +
                                                     abs(segment_stats_df['right_ear_mean'] - cluster_right_mean) +
                                                     abs(segment_stats_df['mar_mean'] - cluster_mar_mean)) / 3)
    segment_stats_df['rating'] = (
        segment_stats_df['data_sufficiency'] * 0.4 +
        segment_stats_df['stability'] * 0.3 +
        segment_stats_df['representativeness'] * 0.3
    )
    
    # Выбираем сегмент с наивысшим рейтингом
    best_segment_idx = segment_stats_df['rating'].idxmax()
    best_segment_info = segment_stats_df.iloc[best_segment_idx]
    
    selected_segment = segments[int(best_segment_info['segment_id']) - 1]
    
    print(f"  Выбран файл для обучения: {best_segment_info['file_name']}")
    print(f"  Рейтинг: {best_segment_info['rating']:.3f}")
    print(f"  Кадров: {best_segment_info['frames']}, "
          f"последовательностей: {best_segment_info['sequences']}")
    print(f"  Средние значения: left_ear={best_segment_info['left_ear_mean']:.4f}, "
          f"right_ear={best_segment_info['right_ear_mean']:.4f}, "
          f"mar={best_segment_info['mar_mean']:.4f}")
    
    # Сохраняем информацию о выбранном сегменте
    cluster_segment_samples[cluster_id] = {
        'segment_id': int(best_segment_info['segment_id']),
        'file_name': best_segment_info['file_name'],
        'frames_count': best_segment_info['frames'],
        'rating': best_segment_info['rating'],
        'left_ear_mean': best_segment_info['left_ear_mean'],
        'right_ear_mean': best_segment_info['right_ear_mean'],
        'mar_mean': best_segment_info['mar_mean'],
        'cluster': cluster_id
    }
    
    # Анализ и предобработка данных
    processed_segment = analyze_and_preprocess_segment(
        selected_segment, feature_columns, feature_columns  # Используем одни и те же колонки для входа и выхода
    )
    
    # Подготавливаем улучшенные данные для LSTM (теперь всего 30 тестовых записей)
    prepared_data = prepare_improved_lstm_data(
        processed_segment,
        feature_columns,
        target_columns,
        sequence_length=10,
        validation_split=0.15,
        test_samples=30  # Фиксируем 30 тестовых записей
    )
    
    if prepared_data is None:
        print(f"  Не удалось подготовить данные для обучения")
        continue
    
    (X_train, y_train, X_val, y_val, X_test, y_test, 
     X_scalers, y_scalers) = prepared_data
    
    # Обучаем модель с комплексной стратегией
    model, strategy_results = train_lstm_with_strategy(
        X_train, y_train, X_val, y_val,
        feature_columns, target_columns,
        cluster_id, best_segment_info['file_name']
    )
    
    # Сохраняем стратегию
    training_strategies[cluster_id] = strategy_results
    
    # Сохранение модели
    model.save(f'lstm_model_cluster_{cluster_id}_improved_3outputs.keras')
    lstm_models[cluster_id] = model
    
    # Сохраняем скалеры
    lstm_scalers_X[cluster_id] = X_scalers
    lstm_scalers_y[cluster_id] = y_scalers
    
    # Прогноз на тестовых данных
    if len(X_test) > 0:
        y_pred_scaled = model.predict(X_test, verbose=0)
        
        # Обратное преобразование масштаба
        y_pred = np.zeros_like(y_pred_scaled)
        y_true_original = np.zeros_like(y_test)
        
        for target_idx in range(y_pred_scaled.shape[1]):
            y_pred[:, target_idx] = y_scalers[target_idx].inverse_transform(
                y_pred_scaled[:, target_idx].reshape(-1, 1)
            ).flatten()
            y_true_original[:, target_idx] = y_scalers[target_idx].inverse_transform(
                y_test[:, target_idx].reshape(-1, 1)
            ).flatten()
        
        # Расчет ошибок для каждого параметра
        metrics_by_param = {}
        for idx, param_name in enumerate(target_columns):
            errors = y_pred[:, idx] - y_true_original[:, idx]
            absolute_errors = np.abs(errors)
            
            # Улучшенный расчет MAPE
            safe_mape = []
            for i in range(len(y_true_original)):
                true_val = y_true_original[i, idx]
                pred_val = y_pred[i, idx]
                
                if abs(true_val) > 1e-10:  # Избегаем деления на очень маленькие числа
                    error_pct = abs((pred_val - true_val) / true_val) * 100
                    safe_mape.append(error_pct)
                else:
                    # Если true значение близко к 0, используем абсолютную ошибку
                    safe_mape.append(abs(pred_val - true_val) * 100)
            
            mape_value = np.mean(safe_mape) if safe_mape else 0
            mae_value = np.mean(absolute_errors)
            rmse_value = np.sqrt(np.mean(errors**2))
            
            metrics_by_param[param_name] = {
                'mape': mape_value,
                'mae': mae_value,
                'rmse': rmse_value
            }
        
        # Общие метрики
        total_mape = np.mean([metrics['mape'] for metrics in metrics_by_param.values()])
        total_mae = np.mean([metrics['mae'] for metrics in metrics_by_param.values()])
        total_rmse = np.mean([metrics['rmse'] for metrics in metrics_by_param.values()])
        
        # R² score для каждого параметра
        from sklearn.metrics import r2_score
        r2_scores = {}
        for idx, param_name in enumerate(target_columns):
            r2_scores[param_name] = r2_score(y_true_original[:, idx], y_pred[:, idx])
        
        lstm_mape_results[cluster_id] = {
            'metrics_by_param': metrics_by_param,
            'total_mape': total_mape,
            'total_mae': total_mae,
            'total_rmse': total_rmse,
            'r2_scores': r2_scores,
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            'test_samples': len(X_test)
        }
        
        print(f"\n  Результаты тестирования для кластера {cluster_id}:")
        for param_name in target_columns:
            print(f"    {param_name}: MAPE={metrics_by_param[param_name]['mape']:.2f}%, "
                  f"MAE={metrics_by_param[param_name]['mae']:.6f}, "
                  f"R²={r2_scores[param_name]:.4f}")
        print(f"    Средние: MAPE={total_mape:.2f}%, MAE={total_mae:.6f}, RMSE={total_rmse:.6f}")
        print(f"    Примеров: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
        
        # Визуализация результатов тестирования для каждого параметра
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        
        for idx, param_name in enumerate(target_columns):
            # График: Прогнозы vs Реальные значения
            axes[idx, 0].plot(y_true_original[:, idx], label='Реальные', marker='o', markersize=3, alpha=0.7)
            axes[idx, 0].plot(y_pred[:, idx], label='Прогнозы', marker='s', markersize=3, alpha=0.7)
            axes[idx, 0].set_title(f'{param_name}: Прогнозы vs Реальные')
            axes[idx, 0].set_xlabel('Пример')
            axes[idx, 0].set_ylabel(param_name)
            axes[idx, 0].legend()
            axes[idx, 0].grid(True, alpha=0.3)
            
            # График: Ошибки
            errors = y_pred[:, idx] - y_true_original[:, idx]
            axes[idx, 1].bar(range(len(errors)), errors, 
                            color=['red' if e > 0 else 'blue' for e in errors], 
                            alpha=0.6)
            axes[idx, 1].axhline(y=0, color='black', linestyle='-', linewidth=1)
            axes[idx, 1].set_title(f'{param_name}: Ошибки прогноза')
            axes[idx, 1].set_xlabel('Пример')
            axes[idx, 1].set_ylabel('Ошибка')
            axes[idx, 1].grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(f'Результаты тестирования LSTM (Кластер {cluster_id})\n{best_segment_info["file_name"][:50]}...', 
                     fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(f'improved_predictions_cluster_{cluster_id}_3outputs.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Прогнозирование для одного сегмента каждого кластера (все 3 параметра)
    print(f"\n  Прогнозирование для выбранного сегмента кластера {cluster_id} (все 3 параметра)...")
    
    # Используем тот же сегмент для прогнозирования
    results_df, metrics, predictions_array, actuals_array = predict_for_single_segment(
        cluster_id, processed_segment, model, X_scalers, y_scalers,
        feature_columns, target_columns, sequence_length=10
    )
    
    if metrics:
        print(f"    Прогнозы для сегмента: {best_segment_info['file_name']}")
        print(f"    Количество прогнозов: {metrics['total']['total_predictions']}")
        for param_name in target_columns:
            print(f"    {param_name}: MAPE={metrics[param_name]['mape']:.2f}%, "
                  f"MAE={metrics[param_name]['mae']:.6f}")
        print(f"    Средние MAPE: {metrics['total']['avg_mape']:.2f}%")
        
        # Сохраняем прогнозы для этого кластера
        segment_predictions[cluster_id] = {
            'segment_id': int(best_segment_info['segment_id']),
            'file_name': best_segment_info['file_name'],
            'predictions_df': results_df,
            'metrics': metrics,
            'predictions_array': predictions_array,
            'actuals_array': actuals_array,
            'model_path': f'lstm_model_cluster_{cluster_id}_improved_3outputs.keras'
        }
        
        # Визуализация прогнозов для всего сегмента (все 3 параметра)
        print(f"  Создание графика прогнозов для всего сегмента (3 параметра)...")
        total_mape, total_mae = visualize_segment_predictions(
            processed_segment.iloc[10:],  # Пропускаем первые 10 точек для последовательностей
            predictions_array,
            actuals_array,
            cluster_id,
            best_segment_info['file_name'],
            target_columns,
            sequence_length=10
        )
        
        # Сохраняем результаты прогнозирования в Excel
        results_df.to_excel(f'segment_predictions_cluster_{cluster_id}_3outputs.xlsx', index=False)
        print(f"    Результаты прогнозирования сохранены в 'segment_predictions_cluster_{cluster_id}_3outputs.xlsx'")
    
    print(f"Модель для кластера {cluster_id} обучена и сохранена")

# ================================================
# СОХРАНЕНИЕ УЛУЧШЕННЫХ РЕЗУЛЬТАТОВ (3 ВЫХОДА)
# ================================================

print("\n" + "="*60)
print("СОХРАНЕНИЕ УЛУЧШЕННЫХ РЕЗУЛЬТАТОВ (3 ВЫХОДА)")
print("="*60)

# Сохраняем информацию о стратегиях обучения
if training_strategies:
    strategies_df = pd.DataFrame()
    for cluster_id, strategies in training_strategies.items():
        for strategy_name, results in strategies.items():
            row = {
                'cluster_id': cluster_id,
                'strategy': strategy_name,
                'val_loss': results['val_loss'],
                'train_loss': results['train_loss'],
                'epochs': results['epochs']
            }
            strategies_df = pd.concat([strategies_df, pd.DataFrame([row])], ignore_index=True)
    
    strategies_df.to_excel('training_strategies_summary_3outputs.xlsx', index=False)
    print("Информация о стратегиях обучения сохранена")

# Сохраняем информацию о прогнозах для каждого кластера
if segment_predictions:
    print("\nСохранение результатов прогнозирования для каждого кластера...")
    
    all_predictions_summary = []
    
    with pd.ExcelWriter('all_segment_predictions_summary_3outputs.xlsx') as writer:
        for cluster_id, pred_info in segment_predictions.items():
            # Сохраняем детальные прогнозы на отдельный лист
            sheet_name = f'Cluster_{cluster_id}'
            if len(sheet_name) > 31:
                sheet_name = sheet_name[:31]
            
            pred_info['predictions_df'].to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Добавляем в сводную таблицу
            summary = {
                'Cluster_ID': cluster_id,
                'File_Name': pred_info['file_name'],
                'Segment_ID': pred_info['segment_id'],
                'Total_Predictions': pred_info['metrics']['total']['total_predictions'],
                'left_ear_MAPE_%': round(pred_info['metrics']['left_ear']['mape'], 2),
                'right_ear_MAPE_%': round(pred_info['metrics']['right_ear']['mape'], 2),
                'mar_MAPE_%': round(pred_info['metrics']['mar']['mape'], 2),
                'Avg_MAPE_%': round(pred_info['metrics']['total']['avg_mape'], 2),
                'Avg_MAE': round(pred_info['metrics']['total']['avg_mae'], 6),
                'Model_Path': pred_info['model_path']
            }
            all_predictions_summary.append(summary)
    
    # Сохраняем сводную таблицу
    summary_df = pd.DataFrame(all_predictions_summary)
    summary_df.to_excel('predictions_summary_3outputs.xlsx', index=False)
    
    print("Результаты прогнозирования сохранены:")
    print(summary_df[['Cluster_ID', 'File_Name', 'Total_Predictions', 
                     'left_ear_MAPE_%', 'right_ear_MAPE_%', 'mar_MAPE_%', 'Avg_MAPE_%']].to_string())

# Сводная таблица результатов обучения
if lstm_mape_results:
    final_summary = []
    for cluster_id, metrics in lstm_mape_results.items():
        training_info = cluster_segment_samples.get(cluster_id, {})
        
        summary = {
            'Cluster_ID': cluster_id,
            'Training_File': training_info.get('file_name', 'N/A'),
            'Files_In_Cluster': len(features_df[features_df['cluster'] == cluster_id]),
            'Training_Samples': metrics.get('training_samples', 0),
            'Validation_Samples': metrics.get('validation_samples', 0),
            'Test_Samples': metrics.get('test_samples', 0),
            'left_ear_MAPE_%': round(metrics['metrics_by_param']['left_ear']['mape'], 2),
            'right_ear_MAPE_%': round(metrics['metrics_by_param']['right_ear']['mape'], 2),
            'mar_MAPE_%': round(metrics['metrics_by_param']['mar']['mape'], 2),
            'Avg_MAPE_%': round(metrics['total_mape'], 2),
            'Avg_MAE': round(metrics['total_mae'], 6),
            'Avg_RMSE': round(metrics['total_rmse'], 6),
            'left_ear_R2': round(metrics['r2_scores']['left_ear'], 4),
            'right_ear_R2': round(metrics['r2_scores']['right_ear'], 4),
            'mar_R2': round(metrics['r2_scores']['mar'], 4),
            'Quality': 'Отличная' if metrics['total_mape'] < 10 else 
                      'Хорошая' if metrics['total_mape'] < 20 else 
                      'Удовлетворительная' if metrics['total_mape'] < 30 else 
                      'Плохая'
        }
        final_summary.append(summary)
    
    final_summary_df = pd.DataFrame(final_summary)
    final_summary_df.to_excel('improved_model_performance_summary_3outputs.xlsx', index=False)
    
    print("\nИтоговые результаты обучения (3 выхода):")
    print(final_summary_df[['Cluster_ID', 'left_ear_MAPE_%', 'right_ear_MAPE_%', 
                           'mar_MAPE_%', 'Avg_MAPE_%', 'Quality']].to_string())
    
    # Визуализация улучшений для каждого параметра
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    clusters = final_summary_df['Cluster_ID'].values
    
    for idx, param_name in enumerate(['left_ear', 'right_ear', 'mar']):
        mape_values = final_summary_df[f'{param_name}_MAPE_%'].values
        
        colors = ['green' if mape < 10 else 'orange' if mape < 20 else 'yellow' if mape < 30 else 'red' 
                  for mape in mape_values]
        
        bars = axes[idx].bar(clusters, mape_values, color=colors, alpha=0.7, edgecolor='black')
        
        axes[idx].axhline(y=10, color='green', linestyle='--', alpha=0.5, label='Отличная (<10%)')
        axes[idx].axhline(y=20, color='orange', linestyle='--', alpha=0.5, label='Хорошая (<20%)')
        axes[idx].axhline(y=30, color='red', linestyle='--', alpha=0.5, label='Приемлемая (<30%)')
        
        for i, (bar, mape) in enumerate(zip(bars, mape_values)):
            axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                          f'{mape:.1f}%', ha='center', va='bottom', fontsize=9)
        
        axes[idx].set_xlabel('Кластер', fontsize=12)
        axes[idx].set_ylabel('MAPE (%)', fontsize=12)
        axes[idx].set_title(f'Точность прогнозирования: {param_name}', fontsize=14)
        axes[idx].legend(loc='upper right')
        axes[idx].grid(True, alpha=0.3, axis='y')
        axes[idx].set_ylim(0, max(mape_values) * 1.2)
    
    plt.suptitle('Улучшенные результаты прогнозирования по кластерам (3 параметра)', fontsize=16, y=1.05)
    plt.tight_layout()
    plt.savefig('improved_performance_summary_3outputs.png', dpi=300, bbox_inches='tight')
    plt.close()

# Создаем единый график со всеми прогнозами для всех параметров
print("\nСоздание единого графика со всеми прогнозами (3 параметра)...")
if segment_predictions:
    n_clusters = len(segment_predictions)
    fig, axes = plt.subplots(n_clusters, 3, figsize=(20, 4 * n_clusters))
    
    if n_clusters == 1:
        axes = np.array([axes]).reshape(1, 3)
    
    for idx, (cluster_id, pred_info) in enumerate(segment_predictions.items()):
        predictions = pred_info['predictions_array']
        actuals = pred_info['actuals_array']
        
        time_indices = np.arange(len(predictions))
        
        # Для каждого параметра создаем свой график
        for param_idx, param_name in enumerate(target_columns):
            actual_values = actuals[:, param_idx]
            predicted_values = predictions[:, param_idx]
            
            axes[idx, param_idx].plot(time_indices, actual_values, label='Фактические', 
                                     color='blue', alpha=0.7, linewidth=1.5)
            axes[idx, param_idx].plot(time_indices, predicted_values, label='Прогнозы', 
                                     color='red', alpha=0.7, linewidth=1.5, linestyle='--')
            
            # Расчет ошибок
            errors = predicted_values - actual_values
            mape = np.mean(np.abs(errors / (actual_values + 1e-10))) * 100
            
            axes[idx, param_idx].text(0.02, 0.95, f'MAPE: {mape:.2f}%', 
                                     transform=axes[idx, param_idx].transAxes,
                                     fontsize=9, verticalalignment='top',
                                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            axes[idx, param_idx].set_xlabel('Временной индекс', fontsize=9)
            axes[idx, param_idx].set_ylabel(param_name, fontsize=9)
            
            if param_idx == 0:
                axes[idx, param_idx].set_title(f'Кластер {cluster_id}\n{param_name}', fontsize=10)
            else:
                axes[idx, param_idx].set_title(param_name, fontsize=10)
            
            axes[idx, param_idx].legend(loc='best', fontsize=8)
            axes[idx, param_idx].grid(True, alpha=0.3)
    
    plt.suptitle('Прогнозы LSTM для всех кластеров и параметров', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('all_segment_predictions_3outputs.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Единый график со всеми прогнозами сохранен как 'all_segment_predictions_3outputs.png'")

print("\n" + "="*60)
print("ОСНОВНЫЕ ИЗМЕНЕНИЯ:")
print("="*60)
print("1. Модель теперь принимает на вход все 3 параметра (left_ear, right_ear, mar)")
print("2. Модель теперь прогнозирует все 3 параметра одновременно")
print("3. Архитектура LSTM адаптирована для многомерного выхода")
print("4. Метрики качества вычисляются для каждого параметра отдельно")
print("5. Визуализации создаются для всех 3 параметров")
print("6. Все модели сохранены с суффиксом '_3outputs'")
print("7. Результаты сохраняются в отдельных файлах Excel для каждого кластера")
print("8. Созданы сводные таблицы по всем параметрам")