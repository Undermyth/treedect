from sklearn.decomposition import PCA  # type: ignore
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import umap
import numpy as np
from typing import List

def cluster(ids: np.ndarray, features: np.ndarray, manual_features: np.ndarray, n_classes: int) -> List[np.ndarray]:
    '''
    ids: ndarray of shape (n_samples,)
    features: ndarray of shape (n_samples, n_features)
    manual_features: ndarray of shape (n_samples, n_manual_features)
    Returns: a list of length n_classes, each element is an ndarray containing the id of the samples in that cluster
    '''

    # Assert that features and manual_features do not contain NaN
    assert not np.isnan(features).any(), "features contain NaN values"
    assert not np.isnan(manual_features).any(), "manual_features contain NaN values"

    # Step 1: PCA降维 - 将features从[N, d]降维到[N, 7]
    n_samples = features.shape[0]
    n_components_pca = min(7, features.shape[1], n_samples)

    if n_components_pca < 7:
        # 如果特征维度或样本数不足，使用最大可用维度
        pca_features = PCA(n_components=n_components_pca).fit_transform(features)
        # 填充到7维
        padding = np.zeros((n_samples, 7 - n_components_pca))
        pca_features = np.concatenate([pca_features, padding], axis=1)
    else:
        pca_features = PCA(n_components=7).fit_transform(features)

    # Step 2: 对manual_features进行normalization，然后拼接到pca_features后面
    scaler_manual = StandardScaler()
    manual_features_normalized = scaler_manual.fit_transform(manual_features)

    # 拼接形成[N, 7+d]的特征
    combined_features = np.concatenate([pca_features, manual_features_normalized], axis=1)

    # 对拼接后的特征进行normalization（可选，但通常有助于后续降维）
    scaler_combined = StandardScaler()
    combined_features_normalized = scaler_combined.fit_transform(combined_features)

    # Step 3: 使用UMAP降维到[N, 4]
    n_components_umap = min(4, combined_features_normalized.shape[1], n_samples - 1)
    if n_components_umap < 4:
        # 如果维度不足，使用最大可用维度并填充
        umap_reducer = umap.UMAP(n_components=n_components_umap, random_state=42)
        umap_features = umap_reducer.fit_transform(combined_features_normalized)
        # 填充到4维
        if n_components_umap < 4:
            padding = np.zeros((n_samples, 4 - n_components_umap))
            umap_features = np.concatenate([umap_features, padding], axis=1)
    else:
        umap_reducer = umap.UMAP(n_components=4, random_state=42)
        umap_features = umap_reducer.fit_transform(combined_features_normalized)

    # Step 4: 使用KMeans聚类
    kmeans = KMeans(n_clusters=n_classes, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(umap_features)

    # 根据聚类结果组织返回数据：返回每个cluster的ids列表
    result: List[np.ndarray] = []
    for i in range(n_classes):
        cluster_ids = ids[cluster_labels == i]
        result.append(cluster_ids)

    return result
