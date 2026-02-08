import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

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

# Входные признаки и целевая переменная
feature_columns = ['left_ear', 'right_ear']
target_columns = ['mar']  # Теперь одна целевая переменная

# ================================================
# ФУНКЦИЯ СГЛАЖИВАНИЯ ДАННЫХ
# ================================================
def smooth_data(df, window_size=3, method='moving_average'):
    """
    Сглаживание данных для уменьшения шума и скачков
    """
    df_smoothed = df.copy()
    
    # Список колонок для сглаживания (все числовые колонки)
    columns_to_smooth = feature_columns + target_columns
    
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

def calculate_temporal_features(segment_df, feature_columns=feature_columns):
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
    
    # Характеристики для целевой переменной mar
    if 'mar' in segment_df.columns:
        mar_data = segment_df['mar'].values
        features['mar_min'] = np.min(mar_data)
        features['mar_max'] = np.max(mar_data)
        features['mar_mean'] = np.mean(mar_data)
        features['mar_median'] = np.median(mar_data)
        features['mar_std'] = np.std(mar_data)
        features['mar_variance'] = np.var(mar_data)
        
        if len(mar_data) > 1:
            features['mar_trend'] = np.polyfit(range(len(mar_data)), mar_data, 1)[0]
            features['mar_autocorr'] = pd.Series(mar_data).autocorr()
            
            # Динамика MAR
            mar_diff = np.diff(mar_data)
            features['mar_diff_mean'] = np.mean(mar_diff)
            features['mar_diff_std'] = np.std(mar_diff)
            features['mar_diff_max'] = np.max(np.abs(mar_diff))
        else:
            features['mar_trend'] = 0
            features['mar_autocorr'] = 0
            features['mar_diff_mean'] = 0
            features['mar_diff_std'] = 0
            features['mar_diff_max'] = 0
        
        # Стабильность MAR
        if np.mean(mar_data) != 0:
            features['mar_cv'] = np.std(mar_data) / np.mean(mar_data)
        else:
            features['mar_cv'] = 0
    
    # Корреляция между признаками
    if len(feature_columns) >= 2 and all(col in segment_df.columns for col in feature_columns):
        if len(segment_df) > 1:
            corr_matrix = segment_df[feature_columns].corr()
            if len(feature_columns) == 2:
                features['corr_left_right'] = corr_matrix.iloc[0, 1]
    
    return features

# Создание DataFrame с характеристиками для каждого сегмента (файла)
print("\n--- Вычисление временных характеристик для каждого файла ---")
features_list = []

for i, segment in enumerate(segments):
    features = calculate_temporal_features(segment, feature_columns)
    features['segment_id'] = i + 1
    features['file_name'] = segment['file_name_original'].iloc[0]
    features['total_frames'] = len(segment)
    features_list.append(features)
    
    if i < 5:  # Показываем информацию о первых 5 файлах
        print(f"  Сегмент {i+1}: {segment['file_name_original'].iloc[0]} - {len(segment)} кадров, "
              f"left_ear: {features['left_ear_mean']:.4f}, right_ear: {features['right_ear_mean']:.4f}")

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
# КЛАСТЕРИЗАЦИЯ С ИСПОЛЬЗОВАНИЕМ KMeans
# ================================================

print("\n--- Кластеризация файлов по временным характеристикам ---")

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
    
    # Определение оптимального числа кластеров методом локтя
    inertia = []
    max_clusters = min(11, len(X_scaled))
    K_range = range(1, max_clusters)
    
    print("Определение оптимального числа кластеров...")
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)
    
    # Автоматическое определение оптимального k (метод локтя)
    if len(inertia) > 2:
        # Вычисляем вторые производные для нахождения "локтя"
        second_derivative = np.diff(np.diff(inertia))
        if len(second_derivative) > 0:
            # Находим точку, где кривизна максимальна
            optimal_k = np.argmax(np.abs(second_derivative)) + 2
        else:
            optimal_k = 3
    else:
        optimal_k = min(3, len(X_scaled))
    
    # Ограничиваем диапазон
    optimal_k = max(2, min(optimal_k, min(5, len(X_scaled)//3)))
    
    print(f"Оптимальное число кластеров: {optimal_k}")
    
    # Визуализация метода локтя
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, inertia, marker='o', linewidth=2, markersize=8)
    plt.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, 
                label=f'Оптимальное k={optimal_k}')
    plt.xlabel('Число кластеров', fontsize=12)
    plt.ylabel('Инерция (Inertia)', fontsize=12)
    plt.title('Метод локтя для определения оптимального числа кластеров', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('elbow_method.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("График метода локтя сохранен как 'elbow_method.png'")
    
    # Выполняем кластеризацию с оптимальным числом кластеров
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    features_df['cluster'] = cluster_labels
    
    # Сохранение результатов кластеризации
    features_df.to_excel('temporal_features_with_clusters.xlsx', index=False)
    
    print("\nРаспределение файлов по кластерам:")
    cluster_distribution = features_df['cluster'].value_counts().sort_index()
    for cluster_id, count in cluster_distribution.items():
        percentage = count / len(features_df) * 100
        avg_frames = features_df[features_df['cluster'] == cluster_id]['total_frames'].mean()
        print(f"  Кластер {cluster_id}: {count} файлов ({percentage:.1f}%), "
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
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                            c=cluster_labels, cmap='tab10', 
                            alpha=0.7, s=100, edgecolors='w', linewidth=1)
        
        # Добавляем центроиды
        if X_scaled.shape[1] > 2:
            centroids_pca = pca.transform(kmeans.cluster_centers_)
        else:
            centroids_pca = kmeans.cluster_centers_
        
        plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
                   c='red', marker='X', s=300, label='Центроиды', 
                   edgecolors='black', linewidth=2)
        
        plt.colorbar(scatter, label='Кластер')
        plt.xlabel('Компонента 1', fontsize=12)
        plt.ylabel('Компонента 2', fontsize=12)
        plt.title(f'Визуализация кластеров файлов ({optimal_k} кластеров)', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Добавляем информацию о файлах (первые несколько)
        for i in range(min(10, len(X_pca))):
            plt.annotate(f"{i+1}", (X_pca[i, 0], X_pca[i, 1]), 
                        fontsize=8, alpha=0.7)
        
        plt.savefig('cluster_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Визуализация кластеров сохранена как 'cluster_visualization.png'")
    except Exception as e:
        print(f"Не удалось создать визуализацию кластеров: {e}")
    
    # Анализ характеристик кластеров
    print("\nАнализ характеристик кластеров:")
    for cluster_id in range(optimal_k):
        cluster_data = features_df[features_df['cluster'] == cluster_id]
        print(f"\nКластер {cluster_id} ({len(cluster_data)} файлов):")
        print(f"  Средняя длина: {cluster_data['total_frames'].mean():.1f} кадров")
        print(f"  Средний left_ear: {cluster_data['left_ear_mean'].mean():.4f}")
        print(f"  Средний right_ear: {cluster_data['right_ear_mean'].mean():.4f}")
        print(f"  Средний mar: {cluster_data['mar_mean'].mean():.4f}")
        
else:
    print("Недостаточно данных для кластеризации")
    features_df['cluster'] = 0
    optimal_k = 1
    kmeans = None
    scaler = None

# ================================================
# ПОДГОТОВКА ДАННЫХ ДЛЯ LSTM
# ================================================

def prepare_lstm_data(segment_df, feature_columns, target_columns, sequence_length=10):
    """
    Подготавливает данные для обучения LSTM из сегмента (файла)
    """
    # Проверяем, что в сегменте достаточно данных
    if len(segment_df) < sequence_length + 1:
        print(f"Предупреждение: сегмент содержит только {len(segment_df)} точек, "
              f"требуется минимум {sequence_length + 1}")
        return np.array([]), np.array([]), [], []
    
    # Извлекаем только нужные колонки
    X_data = segment_df[feature_columns].values
    y_data = segment_df[target_columns].values
    
    X, y = [], []
    
    # Создаем последовательности
    for i in range(len(X_data) - sequence_length):
        X.append(X_data[i:i+sequence_length])
        y.append(y_data[i+sequence_length])
    
    if len(X) == 0:
        return np.array([]), np.array([]), [], []
    
    X = np.array(X)
    y = np.array(y)
    
    # Нормализация данных (отдельно для каждого признака)
    X_scalers = []
    y_scalers = []
    
    # Нормализация признаков
    X_scaled = np.zeros_like(X)
    for feature_idx in range(X.shape[2]):
        feature_scaler = MinMaxScaler()
        feature_data = X[:, :, feature_idx].reshape(-1, 1)
        feature_scaled = feature_scaler.fit_transform(feature_data).reshape(X.shape[0], X.shape[1])
        X_scaled[:, :, feature_idx] = feature_scaled
        X_scalers.append(feature_scaler)
    
    # Нормализация целевых переменных
    y_scaled = np.zeros_like(y)
    for target_idx in range(y.shape[1]):
        target_scaler = MinMaxScaler()
        target_data = y[:, target_idx].reshape(-1, 1)
        target_scaled = target_scaler.fit_transform(target_data).flatten()
        y_scaled[:, target_idx] = target_scaled
        y_scalers.append(target_scaler)
    
    return X_scaled, y_scaled, X_scalers, y_scalers

# ================================================
# СОЗДАНИЕ И ОБУЧЕНИЕ LSTM МОДЕЛЕЙ ДЛЯ КАЖДОГО КЛАСТЕРА
# ================================================

print("\n" + "="*60)
print("ОБУЧЕНИЕ LSTM МОДЕЛЕЙ ДЛЯ ПРОГНОЗИРОВАНИЯ MAR")
print("="*60)

lstm_models = {}
lstm_scalers_X = {}
lstm_scalers_y = {}
lstm_mape_results = {}
lstm_prediction_details = {}

for cluster_id in range(optimal_k):
    print(f"\n--- Обучение LSTM для кластера {cluster_id} ---")
    
    # Получаем сегменты (файлы) текущего кластера
    cluster_segment_ids = features_df[features_df['cluster'] == cluster_id]['segment_id'].values
    cluster_segment_ids = [int(id) - 1 for id in cluster_segment_ids]  # Преобразуем в индексы
    
    if len(cluster_segment_ids) == 0:
        print(f"Кластер {cluster_id} пустой, пропускаем")
        continue
    
    # Собираем все данные из сегментов кластера
    X_all, y_all = [], []
    X_scalers_list, y_scalers_list = [], []
    
    valid_segments = 0
    total_sequences = 0
    
    for seg_id in cluster_segment_ids:
        if seg_id < len(segments):
            segment = segments[seg_id]
            
            # Подготавливаем данные для LSTM
            X_seg, y_seg, X_scalers, y_scalers = prepare_lstm_data(
                segment, 
                feature_columns,
                target_columns,
                sequence_length=10
            )
            
            if len(X_seg) > 0:
                X_all.append(X_seg)
                y_all.append(y_seg)
                X_scalers_list.append(X_scalers)
                y_scalers_list.append(y_scalers)
                valid_segments += 1
                total_sequences += len(X_seg)
    
    if len(X_all) == 0:
        print(f"Нет данных для обучения кластера {cluster_id}")
        continue
    
    # Объединяем данные
    X_all = np.vstack(X_all)
    y_all = np.vstack(y_all)
    
    print(f"Количество файлов в кластере: {valid_segments}")
    print(f"Всего последовательностей: {total_sequences}")
    print(f"Размер данных для обучения: X={X_all.shape}, y={y_all.shape}")
    
    # Создание модели LSTM
    model = Sequential([
        LSTM(64, activation='relu', return_sequences=True, 
             input_shape=(X_all.shape[1], X_all.shape[2])),
        Dropout(0.2),
        LSTM(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(len(target_columns))
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='mse', 
                  metrics=['mae'])
    
    # Обучение модели
    print("Обучение модели LSTM...")
    history = model.fit(
        X_all, y_all,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )
    
    # Сохранение модели
    model.save(f'lstm_model_cluster_{cluster_id}.keras')
    lstm_models[cluster_id] = model
    
    # Сохраняем скалеры из первого файла в кластере
    if X_scalers_list:
        lstm_scalers_X[cluster_id] = X_scalers_list[0]
        lstm_scalers_y[cluster_id] = y_scalers_list[0]
    
    # График потерь при обучении
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title(f'Потери при обучении - Кластер {cluster_id}', fontsize=14)
    plt.xlabel('Эпоха', fontsize=12)
    plt.ylabel('Потери (MSE)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.savefig(f'loss_cluster_{cluster_id}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Прогноз на тестовых данных и расчет ошибок
    if len(X_all) > 10:
        test_idx = min(30, len(X_all))  # Берем до 30 примеров для тестирования
        X_test = X_all[:test_idx]
        y_true = y_all[:test_idx]
        
        y_pred_scaled = model.predict(X_test, verbose=0)
        
        # Обратное преобразование масштаба
        y_pred = np.zeros_like(y_pred_scaled)
        y_true_original = np.zeros_like(y_true)
        
        for target_idx in range(y_pred_scaled.shape[1]):
            y_pred[:, target_idx] = y_scalers_list[0][target_idx].inverse_transform(
                y_pred_scaled[:, target_idx].reshape(-1, 1)
            ).flatten()
            y_true_original[:, target_idx] = y_scalers_list[0][target_idx].inverse_transform(
                y_true[:, target_idx].reshape(-1, 1)
            ).flatten()
        
        # Расчет ошибок
        errors = y_pred - y_true_original
        absolute_errors = np.abs(errors)
        
        # Избегаем деления на ноль при расчете процентных ошибок
        with np.errstate(divide='ignore', invalid='ignore'):
            percentage_errors = np.where(
                y_true_original != 0,
                np.abs(errors) / np.abs(y_true_original) * 100,
                np.abs(errors) * 100  # Если истинное значение равно 0
            )
        
        # Расчет метрик
        mae_values = np.mean(absolute_errors, axis=0)
        mape_values = np.mean(percentage_errors, axis=0)
        rmse_values = np.sqrt(np.mean(errors**2, axis=0))
        
        lstm_mape_results[cluster_id] = {
            'mape': mape_values[0] if len(mape_values) > 0 else 0,
            'mae': mae_values[0] if len(mae_values) > 0 else 0,
            'rmse': rmse_values[0] if len(rmse_values) > 0 else 0
        }
        
        # Сохраняем детали прогнозов
        lstm_prediction_details[cluster_id] = {
            'y_true': y_true_original,
            'y_pred': y_pred,
            'errors': errors,
            'absolute_errors': absolute_errors,
            'percentage_errors': percentage_errors
        }
        
        print(f"Результаты для кластера {cluster_id}:")
        print(f"  MAPE: {mape_values[0]:.2f}%")
        print(f"  MAE: {mae_values[0]:.6f}")
        print(f"  RMSE: {rmse_values[0]:.6f}")
        
        # Графики прогнозов для кластера
        plt.figure(figsize=(15, 10))
        
        # График 1: Прогнозы vs Реальные значения
        plt.subplot(2, 2, 1)
        plt.plot(y_true_original.flatten(), label='Реальные значения', 
                marker='o', linewidth=2, markersize=4, alpha=0.7)
        plt.plot(y_pred.flatten(), label='Прогнозы', 
                marker='s', linewidth=2, markersize=4, alpha=0.7)
        plt.title(f'Прогнозы vs Реальные значения\nКластер {cluster_id}', fontsize=12)
        plt.xlabel('Номер примера')
        plt.ylabel('MAR')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # График 2: Ошибки прогноза
        plt.subplot(2, 2, 2)
        colors = ['green' if e >= 0 else 'red' for e in errors.flatten()]
        plt.bar(range(len(errors.flatten())), errors.flatten(), 
                color=colors, alpha=0.7, edgecolor='black')
        plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
        plt.title(f'Ошибки прогноза\nКластер {cluster_id}', fontsize=12)
        plt.xlabel('Номер примера')
        plt.ylabel('Ошибка (Прогноз - Реальное)')
        plt.grid(True, alpha=0.3, axis='y')
        
        # График 3: Абсолютные ошибки
        plt.subplot(2, 2, 3)
        plt.plot(absolute_errors.flatten(), marker='o', linewidth=2, 
                markersize=4, alpha=0.7, color='orange')
        plt.axhline(y=np.mean(absolute_errors), color='red', 
                   linestyle='--', linewidth=2, 
                   label=f'Среднее: {np.mean(absolute_errors):.6f}')
        plt.title(f'Абсолютные ошибки\nКластер {cluster_id}', fontsize=12)
        plt.xlabel('Номер примера')
        plt.ylabel('Абсолютная ошибка')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # График 4: Процентные ошибки
        plt.subplot(2, 2, 4)
        plt.plot(percentage_errors.flatten(), marker='o', linewidth=2, 
                markersize=4, alpha=0.7, color='purple')
        plt.axhline(y=np.mean(percentage_errors), color='red', 
                   linestyle='--', linewidth=2, 
                   label=f'Среднее: {np.mean(percentage_errors):.2f}%')
        plt.title(f'Процентные ошибки\nКластер {cluster_id}', fontsize=12)
        plt.xlabel('Номер примера')
        plt.ylabel('Ошибка (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'predictions_cluster_{cluster_id}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Гистограмма распределения ошибок
        plt.figure(figsize=(10, 6))
        plt.hist(errors.flatten(), bins=20, alpha=0.7, 
                color='steelblue', edgecolor='black')
        plt.axvline(x=0, color='red', linestyle='--', 
                   linewidth=2, label='Нулевая ошибка')
        plt.axvline(x=np.mean(errors), color='green', linestyle='-', 
                   linewidth=2, label=f'Среднее: {np.mean(errors):.6f}')
        plt.title(f'Распределение ошибок - Кластер {cluster_id}', fontsize=14)
        plt.xlabel('Ошибка (Прогноз - Реальное)')
        plt.ylabel('Частота')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'error_distribution_cluster_{cluster_id}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Детальная таблица прогнозов для кластера
        prediction_table = pd.DataFrame({
            'Sample_Index': range(len(y_true_original)),
            'True_MAR': y_true_original.flatten(),
            'Predicted_MAR': y_pred.flatten(),
            'Error': errors.flatten(),
            'Absolute_Error': absolute_errors.flatten(),
            'Percentage_Error_%': percentage_errors.flatten(),
            'Squared_Error': (errors.flatten())**2
        })
        
        # Добавляем статистику
        stats = {
            'Sample_Index': ['СРЕДНЕЕ', 'СТАНДАРТНОЕ_ОТКЛОНЕНИЕ', 'МИНИМУМ', 'МАКСИМУМ', 'МЕДИАНА', 'MAPE_%'],
            'True_MAR': [np.mean(y_true_original), np.std(y_true_original), 
                        np.min(y_true_original), np.max(y_true_original),
                        np.median(y_true_original), np.nan],
            'Predicted_MAR': [np.mean(y_pred), np.std(y_pred), 
                            np.min(y_pred), np.max(y_pred),
                            np.median(y_pred), np.nan],
            'Error': [np.mean(errors), np.std(errors), 
                     np.min(errors), np.max(errors),
                     np.median(errors), np.nan],
            'Absolute_Error': [np.mean(absolute_errors), np.std(absolute_errors),
                              np.min(absolute_errors), np.max(absolute_errors),
                              np.median(absolute_errors), np.nan],
            'Percentage_Error_%': [np.mean(percentage_errors), np.std(percentage_errors),
                                  np.min(percentage_errors), np.max(percentage_errors),
                                  np.median(percentage_errors), mape_values[0] if len(mape_values) > 0 else 0],
            'Squared_Error': [np.mean((errors.flatten())**2), np.std((errors.flatten())**2),
                             np.min((errors.flatten())**2), np.max((errors.flatten())**2),
                             np.median((errors.flatten())**2), np.nan]
        }
        
        stats_df = pd.DataFrame(stats)
        prediction_table = pd.concat([prediction_table, stats_df], ignore_index=True)
        
        # Сохраняем таблицу
        filename = f'prediction_details_cluster_{cluster_id}.xlsx'
        prediction_table.to_excel(filename, index=False)
        print(f"  Детали прогнозов сохранены в {filename}")
    
    print(f"Модель LSTM для кластера {cluster_id} обучена и сохранена")

# ================================================
# ФУНКЦИЯ ДЛЯ КЛАССИФИКАЦИИ И ПРОГНОЗИРОВАНИЯ НОВЫХ ФАЙЛОВ
# ================================================

def classify_and_predict(new_segment_df, features_df, kmeans_model, scaler, lstm_models, 
                         lstm_scalers_X, lstm_scalers_y, feature_columns, target_columns, 
                         sequence_length=10):
    """
    Классифицирует новый файл и делает прогноз с помощью соответствующей LSTM модели
    """
    # Проверяем, что есть достаточно данных
    if len(new_segment_df) < sequence_length + 1:
        print(f"Ошибка: нужно минимум {sequence_length + 1} точек для прогноза")
        return None, None, None
    
    # 1. Вычисление характеристик для нового файла
    new_features = calculate_temporal_features(new_segment_df, feature_columns)
    
    # 2. Подготовка данных для классификации
    numeric_features_df = features_df.select_dtypes(include=[np.number])
    
    # Исключаем идентификационные колонки и кластер
    exclude_cols = ['segment_id', 'file_name', 'total_frames', 'frames_count', 
                   'duration_frames', 'cluster']
    feature_names = [col for col in numeric_features_df.columns 
                    if col not in exclude_cols]
    
    # Создаем вектор характеристик в том же порядке, что и при обучении
    X_new = np.array([new_features.get(feat, 0) for feat in feature_names]).reshape(1, -1)
    
    # 3. Масштабирование и классификация
    X_new_scaled = scaler.transform(X_new)
    cluster_label = kmeans_model.predict(X_new_scaled)[0]
    
    print(f"Новый файл отнесен к кластеру: {cluster_label}")
    
    # 4. Проверка наличия модели для этого кластера
    if cluster_label not in lstm_models:
        print(f"Модель для кластера {cluster_label} не найдена")
        return None, None, cluster_label
    
    # 5. Подготовка данных для LSTM
    # Берем последние sequence_length точек
    X_data = new_segment_df[feature_columns].values[-sequence_length:]
    
    # Масштабирование признаков
    X_scaled = np.zeros((1, sequence_length, len(feature_columns)))
    for feature_idx in range(len(feature_columns)):
        feature_scaler = lstm_scalers_X[cluster_label][feature_idx]
        feature_data = X_data[:, feature_idx].reshape(-1, 1)
        feature_scaled = feature_scaler.transform(feature_data).flatten()
        X_scaled[0, :, feature_idx] = feature_scaled
    
    # 6. Прогнозирование
    y_pred_scaled = lstm_models[cluster_label].predict(X_scaled, verbose=0)
    
    # Обратное преобразование масштаба
    y_pred = np.zeros(len(target_columns))
    for target_idx in range(len(target_columns)):
        y_pred[target_idx] = lstm_scalers_y[cluster_label][target_idx].inverse_transform(
            y_pred_scaled[:, target_idx].reshape(-1, 1)
        ).flatten()[0]
    
    # Получаем последние известные значения признаков
    last_features = new_segment_df[feature_columns].iloc[-1].values
    
    return y_pred, last_features, cluster_label

# ================================================
# СОЗДАНИЕ ОТЧЕТОВ И ВИЗУАЛИЗАЦИЙ
# ================================================

print("\n" + "="*60)
print("СОЗДАНИЕ ОТЧЕТОВ")
print("="*60)

# 1. Сводный отчет по кластерам
if lstm_mape_results:
    # График метрик по кластерам
    plt.figure(figsize=(12, 6))
    
    clusters = list(lstm_mape_results.keys())
    mape_values = [lstm_mape_results[c]['mape'] for c in clusters]
    mae_values = [lstm_mape_results[c]['mae'] for c in clusters]
    rmse_values = [lstm_mape_results[c]['rmse'] for c in clusters]
    
    x = np.arange(len(clusters))
    width = 0.25
    
    plt.bar(x - width, mape_values, width, label='MAPE (%)', color='skyblue', alpha=0.8)
    plt.bar(x, mae_values, width, label='MAE', color='lightgreen', alpha=0.8)
    plt.bar(x + width, rmse_values, width, label='RMSE', color='salmon', alpha=0.8)
    
    # Добавляем значения на столбцы
    for i, (mape, mae, rmse) in enumerate(zip(mape_values, mae_values, rmse_values)):
        plt.text(i - width, mape + 0.5, f'{mape:.1f}%', ha='center', va='bottom', fontsize=9)
        plt.text(i, mae + 0.0001, f'{mae:.4f}', ha='center', va='bottom', fontsize=9)
        plt.text(i + width, rmse + 0.0001, f'{rmse:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.xlabel('ID кластера', fontsize=12)
    plt.ylabel('Метрика ошибки', fontsize=12)
    plt.title('Метрики производительности моделей по кластерам', fontsize=14, pad=20)
    plt.xticks(x, [f'Кластер {c}' for c in clusters])
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('model_performance_by_cluster.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("График метрик производительности сохранен как 'model_performance_by_cluster.png'")
    
    # Создаем сводную таблицу метрик
    metrics_summary = []
    for cluster_id in clusters:
        metrics = lstm_mape_results[cluster_id]
        cluster_files = features_df[features_df['cluster'] == cluster_id]
        
        metrics_summary.append({
            'Cluster_ID': cluster_id,
            'Files_Count': len(cluster_files),
            'Avg_Frames_Per_File': cluster_files['total_frames'].mean(),
            'MAPE_%': round(metrics['mape'], 2),
            'MAE': round(metrics['mae'], 6),
            'RMSE': round(metrics['rmse'], 6),
            'LSTM_Model': 'Да' if cluster_id in lstm_models else 'Нет'
        })
    
    metrics_df = pd.DataFrame(metrics_summary)
    metrics_df.to_excel('model_metrics_summary.xlsx', index=False)
    print("Сводная таблица метрик сохранена в 'model_metrics_summary.xlsx'")

# 2. Общий отчет
report_data = {
    'Параметр': [
        'Всего файлов',
        'Всего кадров',
        'Входные признаки',
        'Целевая переменная',
        'Число кластеров',
        'Моделей LSTM обучено',
        'Длина последовательности LSTM',
        'Метод сглаживания',
        'Размер окна сглаживания',
        'Средняя длина файла (кадры)',
        'Минимальная длина файла',
        'Максимальная длина файла'
    ],
    'Значение': [
        len(segments),
        len(df_smoothed),
        ', '.join(feature_columns),
        target_columns[0],
        optimal_k,
        len(lstm_models),
        10,
        'скользящее среднее',
        3,
        f"{np.mean([len(s) for s in segments]):.1f}",
        min([len(s) for s in segments]),
        max([len(s) for s in segments])
    ]
}

report_df = pd.DataFrame(report_data)

# 3. Статистика по кластерам
if 'cluster' in features_df.columns:
    cluster_stats = features_df['cluster'].value_counts().sort_index()
    cluster_report = []
    
    for cluster_id, count in cluster_stats.items():
        cluster_data = features_df[features_df['cluster'] == cluster_id]
        mape = lstm_mape_results.get(cluster_id, {}).get('mape', 'N/A')
        
        cluster_report.append({
            'Cluster_ID': cluster_id,
            'Files_Count': count,
            'Percentage_%': round(count / len(features_df) * 100, 1),
            'Avg_Frames': round(cluster_data['total_frames'].mean(), 1),
            'Avg_left_ear': round(cluster_data['left_ear_mean'].mean(), 4),
            'Avg_right_ear': round(cluster_data['right_ear_mean'].mean(), 4),
            'Avg_mar': round(cluster_data['mar_mean'].mean(), 4),
            'Avg_MAPE_%': round(mape, 2) if mape != 'N/A' else 'N/A',
            'Model_Trained': 'Да' if cluster_id in lstm_models else 'Нет'
        })
    
    cluster_stats_df = pd.DataFrame(cluster_report)
else:
    cluster_stats_df = pd.DataFrame({'Сообщение': ['Кластеризация не выполнялась']})

# Сохранение полного отчета
with pd.ExcelWriter('analysis_report.xlsx') as writer:
    report_df.to_excel(writer, sheet_name='Итоговый_отчет', index=False)
    cluster_stats_df.to_excel(writer, sheet_name='Статистика_по_кластерам', index=False)
    features_df.to_excel(writer, sheet_name='Характеристики_файлов', index=False)
    
    if lstm_mape_results:
        metrics_df.to_excel(writer, sheet_name='Метрики_моделей', index=False)

print("Полный отчет сохранен в 'analysis_report.xlsx'")

# ================================================
# ДЕМОНСТРАЦИЯ РАБОТЫ АЛГОРИТМА
# ================================================

print("\n" + "="*60)
print("ДЕМОНСТРАЦИЯ РАБОТЫ АЛГОРИТМА")
print("="*60)

if len(segments) > 0 and kmeans is not None:
    # Берем случайный файл для демонстрации
    demo_segment_idx = np.random.randint(0, len(segments))
    demo_segment = segments[demo_segment_idx]
    demo_file_name = demo_segment['file_name_original'].iloc[0]
    
    print(f"Демонстрация на файле: {demo_file_name}")
    print(f"Количество кадров: {len(demo_segment)}")
    print(f"Диапазон кадров: {demo_segment['frame_number'].min()} - {demo_segment['frame_number'].max()}")
    
    # Классифицируем и прогнозируем
    prediction, last_features, cluster = classify_and_predict(
        demo_segment, features_df, kmeans, scaler, lstm_models,
        lstm_scalers_X, lstm_scalers_y, feature_columns, target_columns
    )
    
    if prediction is not None:
        print(f"\nРезультаты демонстрации:")
        print(f"  Отнесен к кластеру: {cluster}")
        print(f"  Последние известные значения признаков:")
        for i, col in enumerate(feature_columns):
            print(f"    {col}: {last_features[i]:.6f}")
        print(f"  Прогнозируемое значение MAR: {prediction[0]:.6f}")
        
        # Сравниваем с реальным следующим значением (если есть)
        if len(demo_segment) > 10:
            # Берем значение после последней точки в последовательности
            next_real_idx = 10  # sequence_length
            if next_real_idx < len(demo_segment):
                real_next_value = demo_segment['mar'].iloc[next_real_idx]
                error = prediction[0] - real_next_value
                percentage_error = abs(error) / abs(real_next_value) * 100 if real_next_value != 0 else abs(error) * 100
                
                print(f"  Реальное следующее значение MAR: {real_next_value:.6f}")
                print(f"  Ошибка прогноза: {error:.6f}")
                print(f"  Процентная ошибка: {percentage_error:.2f}%")
else:
    print("Недостаточно данных для демонстрации")

# ================================================
# СОХРАНЕНИЕ ВСЕХ РЕЗУЛЬТАТОВ
# ================================================

# Сохранение сглаженных данных
df_smoothed.to_excel('smoothed_dataset.xlsx', index=False)
print("\nСглаженные данные сохранены в 'smoothed_dataset.xlsx'")

# Создание README файла с описанием
readme_content = f"""
АНАЛИЗ ВРЕМЕННЫХ РЯДОВ ДЛЯ ПРОГНОЗИРОВАНИЯ MAR

СТРУКТУРА ДАННЫХ:
- Каждый файл (file_name) = отдельный сегмент
- Входные признаки: left_ear, right_ear
- Целевая переменная: mar
- Frame_number используется только для отслеживания порядка

АНАЛИЗ ВЫПОЛНЕН:
- Обработано файлов: {len(segments)}
- Общее количество кадров: {len(df_smoothed)}
- Средняя длина файла: {np.mean([len(s) for s in segments]):.1f} кадров
- Кластеризация: {optimal_k} кластера(ов)
- Обучено LSTM моделей: {len(lstm_models)}

ЭТАПЫ АНАЛИЗА:
1. Сглаживание данных (скользящее среднее, окно=3)
2. Каждый файл = отдельный сегмент
3. Извлечение временных характеристик для каждого файла
4. Кластеризация файлов по характеристикам (KMeans)
5. Обучение отдельной LSTM модели для каждого кластера
6. Оценка качества прогнозирования (MAPE, MAE, RMSE)

РЕЗУЛЬТАТЫ:
- Метрики качества сохранены в model_metrics_summary.xlsx
- Все графики и таблицы в соответствующих файлах
- Модели LSTM сохранены как lstm_model_cluster_*.keras

ИСПОЛЬЗОВАНИЕ:
Для прогнозирования нового файла используйте функцию classify_and_predict()
"""
with open('README.txt', 'w', encoding='utf-8') as f:
    f.write(readme_content)

print("\n" + "="*60)
print("СОЗДАННЫЕ ФАЙЛЫ")
print("="*60)

files_by_category = {
    "Исходные и обработанные данные": [
        "smoothed_dataset.xlsx",
        "segment_info.xlsx",
        "segmented_time_series.xlsx"
    ],
    "Характеристики и кластеризация": [
        "temporal_features.xlsx",
        "temporal_features_with_clusters.xlsx",
        "segment_lengths_distribution.png",
        "elbow_method.png",
        "cluster_visualization.png"
    ],
    "Модели LSTM": [f"lstm_model_cluster_{cid}.keras" for cid in lstm_models.keys()],
    "Отчеты и метрики": [
        "analysis_report.xlsx",
        "model_metrics_summary.xlsx",
        "model_performance_by_cluster.png",
        "README.txt"
    ],
    "Графики обучения LSTM": [f"loss_cluster_{cid}.png" for cid in lstm_models.keys()],
    "Графики прогнозов LSTM": [f"predictions_cluster_{cid}.png" for cid in lstm_models.keys()],
    "Графики ошибок LSTM": [f"error_distribution_cluster_{cid}.png" for cid in lstm_models.keys()],
    "Таблицы прогнозов LSTM": [f"prediction_details_cluster_{cid}.xlsx" for cid in lstm_models.keys()]
}

for category, files in files_by_category.items():
    if files:
        print(f"\n{category}:")
        for file in files[:5]:  # Показываем первые 5 файлов
            print(f"  - {file}")
        if len(files) > 5:
            print(f"  ... и еще {len(files) - 5} файлов")

print("\n" + "="*60)
print("СКРИПТ УСПЕШНО ВЫПОЛНЕН!")
print("="*60)
print(f"Обработано {len(segments)} файлов (сегментов)")
print(f"Общее количество кадров: {len(df_smoothed)}")
print(f"Создано {len(lstm_models)} LSTM моделей для {optimal_k} кластеров")
print(f"Средний MAPE по кластерам: {np.mean([lstm_mape_results[c]['mape'] for c in lstm_mape_results.keys()]):.2f}%" if lstm_mape_results else "Метрики не рассчитаны")