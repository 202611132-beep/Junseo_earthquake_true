import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import folium
from streamlit_folium import st_folium

st.set_page_config(
    page_title="세계 지진 데이터 분석 및 예측",
    page_icon="🌍",
    layout="wide"
)

# CSS 스타일
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1a1a2e;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        color: #555;
        margin-bottom: 1.5rem;
        font-size: 0.95rem;
    }
    .risk-high {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
    }
    .risk-low {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #e8f4fd;
        border: 1px solid #bee5eb;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        color: #2c6e9e;
    }
    .stButton>button {
        background-color: #e74c3c;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1.5rem;
        font-size: 1rem;
        font-weight: bold;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #c0392b;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def fetch_earthquake_data(lat, lon, radius_deg=5):
    """USGS API에서 지진 데이터 가져오기"""
    min_lat = lat - radius_deg
    max_lat = lat + radius_deg
    min_lon = lon - radius_deg
    max_lon = lon + radius_deg

    # 최근 10년치 데이터
    end_time = datetime.utcnow().strftime("%Y-%m-%d")
    start_time = (datetime.utcnow() - timedelta(days=365 * 10)).strftime("%Y-%m-%d")

    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    params = {
        "format": "geojson",
        "starttime": start_time,
        "endtime": end_time,
        "minlatitude": min_lat,
        "maxlatitude": max_lat,
        "minlongitude": min_lon,
        "maxlongitude": max_lon,
        "minmagnitude": 2.0,
        "orderby": "time",
        "limit": 1000
    }

    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        earthquakes = []
        for feature in data.get("features", []):
            props = feature["properties"]
            coords = feature["geometry"]["coordinates"]
            earthquakes.append({
                "magnitude": props.get("mag", 0),
                "place": props.get("place", "Unknown"),
                "time": datetime.utcfromtimestamp(props["time"] / 1000),
                "longitude": coords[0],
                "latitude": coords[1],
                "depth": coords[2]
            })

        return pd.DataFrame(earthquakes)
    except Exception as e:
        st.error(f"데이터를 가져오는 중 오류가 발생했습니다: {e}")
        return pd.DataFrame()


def predict_risk(df):
    """지진 위험도 예측"""
    if df.empty:
        return "낮음", 0

    count = len(df)
    avg_mag = df["magnitude"].mean()
    max_mag = df["magnitude"].max()
    recent_30d = df[df["time"] > datetime.utcnow() - timedelta(days=30)]
    recent_count = len(recent_30d)

    # 위험도 점수 계산
    score = 0
    score += min(count / 10, 30)         # 총 건수 (최대 30점)
    score += min(avg_mag * 5, 30)        # 평균 규모 (최대 30점)
    score += min(max_mag * 3, 20)        # 최대 규모 (최대 20점)
    score += min(recent_count * 2, 20)   # 최근 활동 (최대 20점)

    if score >= 60:
        return "높음", score
    elif score >= 30:
        return "보통", score
    else:
        return "낮음", score


def create_map(lat, lon, df):
    """Folium 지도 생성"""
    m = folium.Map(location=[lat, lon], zoom_start=6, tiles="CartoDB positron")

    # 중심 마커 (별표)
    folium.Marker(
        location=[lat, lon],
        icon=folium.DivIcon(
            html='<div style="font-size:20px;">⭐</div>',
            icon_size=(25, 25),
            icon_anchor=(12, 12)
        ),
        popup="분석 위치"
    ).add_to(m)

    # 지진 데이터 마커
    if not df.empty:
        for _, row in df.iterrows():
            mag = row["magnitude"]
            color = "red" if mag >= 4.0 else "blue"
            radius = max(3, mag * 2)

            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=radius,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.6,
                popup=folium.Popup(
                    f"규모: {mag:.1f}<br>위치: {row['place']}<br>날짜: {row['time'].strftime('%Y-%m-%d')}",
                    max_width=250
                )
            ).add_to(m)

    return m


# ─── UI 시작 ─────────────────────────────────────────────────
st.markdown('<div class="main-title">🌍 세계 지진 데이터 분석 및 예측</div>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">특정 위치의 위도와 경도를 입력하면, 주변 5도 이내의 과거 지진 데이터를 분석하여 <strong>예상 위험도</strong>를 알려줍니다.</p>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)
with col1:
    lat = st.number_input("위도 입력 (Latitude)", value=36.0, min_value=-90.0, max_value=90.0, step=0.1, format="%.4f")
with col2:
    lon = st.number_input("경도 입력 (Longitude)", value=128.0, min_value=-180.0, max_value=180.0, step=0.1, format="%.4f")

predict_btn = st.button("위험도 예측하기")

if predict_btn:
    with st.spinner("지진 데이터를 불러오는 중..."):
        df = fetch_earthquake_data(lat, lon, radius_deg=5)

    if df is not None and not df.empty:
        risk_level, score = predict_risk(df)
        count = len(df)

        # 위험도 결과
        if risk_level == "높음":
            icon = "🔔"
            box_class = "risk-high"
        elif risk_level == "보통":
            icon = "⚠️"
            box_class = "risk-low"
        else:
            icon = "✅"
            box_class = "risk-low"

        st.markdown(
            f'<div class="{box_class}"><h3>{icon} 예측된 위험도: {risk_level}</h3></div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="info-box">주변 5도 이내에서 총 <strong>{count}건</strong>의 지진 데이터를 분석한 결과입니다.</div>',
            unsafe_allow_html=True
        )

        # 지도
        m = create_map(lat, lon, df)
        st_folium(m, width=700, height=400)

        # 통계 정보
        st.subheader("📊 통계 요약")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("총 지진 건수", f"{count}건")
        col2.metric("평균 규모", f"{df['magnitude'].mean():.2f}")
        col3.metric("최대 규모", f"{df['magnitude'].max():.1f}")
        col4.metric("최근 30일", f"{len(df[df['time'] > datetime.utcnow() - timedelta(days=30)])}건")

        # 규모 분포
        st.subheader("📈 규모별 분포")
        bins = [2, 3, 4, 5, 6, 10]
        labels = ["2~3", "3~4", "4~5", "5~6", "6+"]
        df["mag_range"] = pd.cut(df["magnitude"], bins=bins, labels=labels)
        dist = df["mag_range"].value_counts().sort_index()
        st.bar_chart(dist)

        # 최근 지진 목록
        st.subheader("🗂 최근 지진 목록 (최대 20건)")
        recent_df = df.sort_values("time", ascending=False).head(20)[["time", "magnitude", "depth", "place"]]
        recent_df.columns = ["발생 시간", "규모", "깊이(km)", "위치"]
        recent_df["발생 시간"] = recent_df["발생 시간"].dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(recent_df, use_container_width=True)

    else:
        st.markdown(
            '<div class="risk-low"><h3>✅ 예측된 위험도: 낮음</h3></div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="info-box">주변 5도 이내에서 유의미한 지진 데이터가 없습니다.</div>',
            unsafe_allow_html=True
        )
        m = create_map(lat, lon, pd.DataFrame())
        st_folium(m, width=700, height=400)
