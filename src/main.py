import os
import time
import numpy as np
import mne
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB

# 분리한 모듈 임포트
from WSDF import OlfactoryEEGClassifier
from channel_selection import select_channels_by_correlation
from visualization import plot_confusion_matrix

def run_cross_validation(data_dir, n_folds=5, correlation_threshold=1.0, random_state_cv=42):
    # 1. 초기 데이터 로드 및 채널 정보 확인
    print("Step 1: Initializing for channel selection")
    init_clf = OlfactoryEEGClassifier(data_dir=data_dir, random_state=random_state_cv)
    try:
        X_raw, _, _ = init_clf.load_data() # 전체 데이터 로드 시도
    except Exception as e:
        print(f"Failed to load data for selection: {e}")
        return

    # 채널 이름 확보
    channels = init_clf.all_channel_names_loaded_
    if not channels:
         # fallback: 첫 번째 파일에서 읽기
        first_file = next((f for f in os.listdir(data_dir) if f.endswith('.set')), None)
        if first_file:
            raw_info = mne.io.read_raw_eeglab(os.path.join(data_dir, first_file), preload=False, verbose=False)
            channels = raw_info.ch_names

    # 2. 상관계수 기반 채널 선택
    print("Step 2: Selecting channels by correlation")
    sel_idx, sel_names = select_channels_by_correlation(X_raw, channels, threshold=correlation_threshold, verbose=False)
    print(f"Selected {len(sel_idx)} channels: {sel_names}")

    # 3. 선택된 채널로 메인 Classifier 인스턴스 생성
    clf = OlfactoryEEGClassifier(data_dir, selected_channel_indices=sel_idx, random_state=random_state_cv)
    X, y, _ = clf.load_data()

    # 모델 정의
    classifiers = {
        'SVM': SVC(kernel='linear', C=0.5, random_state=random_state_cv),
        'KNN': KNeighborsClassifier(n_neighbors=1),
        # 'NB': GaussianNB()
    }
    
    # 결과 저장용 변수
    results = {name: {'acc': [], 'time': [], 'y_true': [], 'y_pred': []} for name in classifiers}

    # CV 시작
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state_cv)
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y), 1):
        print(f"\n--- Fold {fold}/{n_folds} ---")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 특징 추출
        X_train_feats, y_train_proc, _ = clf.extract_features(X_train, y_train, is_training=True)
        X_test_feats, y_test_proc, _ = clf.extract_features(X_test, y_test, clf.spatial_filter, is_training=False)

        for name, model in classifiers.items():
            t0 = time.time()
            model.fit(X_train_feats, y_train_proc)
            fit_time = time.time() - t0
            
            y_pred = model.predict(X_test_feats)
            acc = accuracy_score(y_test_proc, y_pred)
            
            results[name]['acc'].append(acc)
            results[name]['time'].append(fit_time)
            results[name]['y_true'].extend(y_test_proc)
            results[name]['y_pred'].extend(y_pred)
            
            print(f"{name}: Accuracy={acc:.4f}, Time={fit_time:.2f}s")

    # 4. 최종 결과 출력 및 시각화
    print("\n--- Final Performance Evaluation ---")
    for name in classifiers:
        mean_acc = np.mean(results[name]['acc'])
        print(f"{name}: Mean Accuracy = {mean_acc:.4f}")
        
        # 시각화 모듈 호출
        plot_confusion_matrix(
            results[name]['y_true'], 
            results[name]['y_pred'], 
            name, 
            clf.classes
        )

if __name__ == "__main__":
    # 경로 설정
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(BASE_DIR, "data") # data 폴더 하위로 가정
    
    if os.path.isdir(data_path):
        run_cross_validation(data_path, correlation_threshold=0.98) # correlation_threshold 조정
    else:
        print(f"Error: Data directory not found at {data_path}")