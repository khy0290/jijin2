import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

# --- 1. Random Forest 모델 학습 함수 ---
@st.cache_resource # 모델을 캐싱하여 파일 업로드마다 재학습하는 것을 방지
def train_random_forest_model(df):
    """
    업로드된 데이터로 Random Forest 모델을 학습시키고 결과를 반환합니다.
    """
    # 1. 특징(Features) 및 레이블(Label) 정의
    # 쓰나미 예측에 사용될 입력 변수: distance_to_coast 제외
    FEATURES = ['magnitude', 'depth', 'latitude', 'longitude']
    # 예측할 출력 변수 (0: 미발생, 1: 발생)
    LABEL = 'tsunami'
    
    # 필수 열이 데이터에 포함되어 있는지 확인
    missing_cols = [col for col in FEATURES + [LABEL] if col not in df.columns]
    if missing_cols:
        st.error(f"필수 열이 데이터에 없습니다: {', '.join(missing_cols)}")
        return None, None, None, None, None

    # 데이터 분리
    X = df[FEATURES]
    y = df[LABEL]

    # 학습 데이터와 테스트 데이터 분리 (예시: 80% 학습, 20% 테스트)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 2. Random Forest 모델 초기화 및 학습
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    # 3. 모델 성능 평가 (테스트 데이터 사용)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    return model, accuracy, report, FEATURES, X

# --- 2. 쓰나미 경보 및 대피 요령 ---

def display_tsunami_warning(df_results):
    """
    예측된 쓰나미 발생 확률에 따라 경고 및 대피 요령을 표시합니다.
    """
    st.subheader("🚨 예측된 쓰나미 위험 지수 및 경보")
    
    # 1. 평균 위험 지수 계산
    avg_probability = df_results['Tsunami Probability (%)'].mean()
    
    st.metric(label="전체 데이터셋 평균 쓰나미 위험 지수", value=f"{avg_probability:.2f}%", delta_color="off")
    
    # 2. 위험 레벨에 따른 경고
    
    if avg_probability >= 50:
        st.error("### 🔴 **높은 위험 감지!**")
        st.warning("**즉시 경계 태세**를 갖추고, 해당 지역의 **가장 높은 곳**으로 이동할 준비를 하십시오. 공식 경보를 주시하세요.")
    elif avg_probability >= 25:
        st.warning("### 🟠 **중간 위험 감지!**")
        st.info("쓰나미 발생 가능성이 있으니, 해안가 근처에서는 **경계**하고 대피 계획을 확인하십시오.")
    else:
        st.success("### 🟢 **낮은 위험 감지!**")
        st.caption("현재 데이터 기준으로는 위험이 낮게 예측됩니다. 하지만 강한 지진 발생 시 항상 주의하십시오.")

    st.markdown("---")
