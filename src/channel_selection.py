import numpy as np

def select_channels_by_correlation(X_all_data, all_channel_names, threshold, verbose=True):
    """
    EEG 채널 간의 상관 관계를 기반으로 중복되는 채널을 제거합니다.
    """
    if X_all_data.ndim != 3 or X_all_data.shape[1] < 2:
        print("Warning: Not enough data or channels. Using all channels.")
        return list(range(X_all_data.shape[1])), all_channel_names

    n_samples, n_channels, n_times = X_all_data.shape

    # 채널 이름 예외 처리
    if not all_channel_names or len(all_channel_names) != n_channels:
        all_channel_names = [f"Ch{i + 1}" for i in range(n_channels)]

    # 데이터 flatten 및 상관계수 계산
    data_for_corr = X_all_data.transpose(1, 0, 2).reshape(n_channels, -1)
    corr_matrix = np.corrcoef(data_for_corr)

    if verbose:
        print(f"\nChecking channel pairs with |corr| > {threshold}:")

    channels_to_remove_indices = []

    # 상관계수가 높은 채널 중 뒤쪽 인덱스를 제거 대상으로 선정
    for i in range(n_channels):
        if i in channels_to_remove_indices:
            continue
        for j in range(i + 1, n_channels):
            if j in channels_to_remove_indices:
                continue
            if abs(corr_matrix[i, j]) > threshold:
                channels_to_remove_indices.append(j)
                if verbose:
                    print(f"  Removing '{all_channel_names[j]}' (correlated with '{all_channel_names[i]}')")

    final_selected_indices = [idx for idx in range(n_channels) if idx not in channels_to_remove_indices]
    final_selected_names = [all_channel_names[i] for i in final_selected_indices]

    if verbose:
        print(f"Selected {len(final_selected_indices)} / {n_channels} channels.")

    if not final_selected_indices:
        return [0], [all_channel_names[0]]

    return final_selected_indices, final_selected_names