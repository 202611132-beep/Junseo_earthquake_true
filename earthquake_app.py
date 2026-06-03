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
    .risk-medium {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
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
    }
    .stButton>button:hover {
        background-color: #c0392b;
    }
</style>
""", unsafe_allow_html=True)


# ── session_state 초기화 ──────────────────────────────────────
for key in ["df", "risk_level", "score", "result_lat", "result_lon"]:
    if key not in st.session_state:
        st.session_state[key] = None


@st.cache_data(ttl=3600)
def fetch_earthquake_data(lat, lon, radius_deg=5):
    min_lat = lat - radius_deg
    max_lat = lat + radius_deg
    min_lon = lon - radius_deg
    max_lon = lon + radius_deg

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
        response = requests.get(url, params=params, timeout=20)
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
    if df is None or df.empty:
        return "낮음", 0

    count = len(df)
    avg_mag = df["magnitude"].mean()
    max_mag = df["magnitude"].max()
    recent_30d = df[df["time"] > datetime.utcnow() - timedelta(days=30)]
    recent_count = len(recent_30d)

    score = 0
    score += min(count / 10, 30)
    score += min(avg_mag * 5, 30)
    score += min(max_mag * 3, 20)
    score += min(recent_count * 2, 20)

    if score >= 60:
        return "높음", score
    elif score >= 30:
        return "보통", score
    else:
        return "낮음", score


@st.cache_data(ttl=3600)
def build_map_html(lat, lon, df_json, df_empty):
    """지도를 HTML 문자열로 변환 — 캐싱 가능하고 재렌더링 안정적"""
    m = folium.Map(location=[lat, lon], zoom_start=6, tiles="CartoDB positron")

    folium.Marker(
        location=[lat, lon],
        icon=folium.DivIcon(
            html='<div style="font-size:22px; text-shadow:0 0 4px #fff;">⭐</div>',
            icon_size=(28, 28),
            icon_anchor=(14, 14)
        ),
        popup="분석 위치"
    ).add_to(m)

    if not df_empty:
        df = pd.read_json(df_json)
        # time 컬럼 복원
        df["time"] = pd.to_datetime(df["time"])
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

    return m._repr_html_()


# ── UI ────────────────────────────────────────────────────────
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

if st.button("위험도 예측하기"):
    with st.spinner("지진 데이터를 불러오는 중..."):
        df = fetch_earthquake_data(lat, lon, radius_deg=5)
    risk_level, score = predict_risk(df)
    # 결과를 session_state에 저장 → 이후 리런에도 유지됨
    st.session_state.df = df
    st.session_state.risk_level = risk_level
    st.session_state.score = score
    st.session_state.result_lat = lat
    st.session_state.result_lon = lon

# ── 결과 표시 (session_state 기반 → 리런해도 유지) ──────────
if st.session_state.risk_level is not None:
    df = st.session_state.df
    risk_level = st.session_state.risk_level
    result_lat = st.session_state.result_lat
    result_lon = st.session_state.result_lon

    if risk_level == "높음":
        icon, box_class = "🔔", "risk-high"
    elif risk_level == "보통":
        icon, box_class = "⚠️", "risk-medium"
    else:
        icon, box_class = "✅", "risk-low"

    st.markdown(
        f'<div class="{box_class}"><h3>{icon} 예측된 위험도: {risk_level}</h3></div>',
        unsafe_allow_html=True
    )

    count = len(df) if (df is not None and not df.empty) else 0
    st.markdown(
        f'<div class="info-box">주변 5도 이내에서 총 <strong>{count}건</strong>의 지진 데이터를 분석한 결과입니다.</div>',
        unsafe_allow_html=True
    )

    # 지도: folium HTML을 components로 렌더 → st_folium보다 안정적
    df_empty = df is None or df.empty
    df_json = df.to_json() if not df_empty else "{}"
    map_html = build_map_html(result_lat, result_lon, df_json, df_empty)
    st.components.v1.html(map_html, height=430, scrolling=False)

    if not df_empty:
        st.subheader("📊 통계 요약")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("총 지진 건수", f"{count}건")
        c2.metric("평균 규모", f"{df['magnitude'].mean():.2f}")
        c3.metric("최대 규모", f"{df['magnitude'].max():.1f}")
        recent_n = len(df[df["time"] > datetime.utcnow() - timedelta(days=30)])
        c4.metric("최근 30일", f"{recent_n}건")

        st.subheader("📈 규모별 분포")
        bins = [2, 3, 4, 5, 6, 10]
        labels = ["2~3", "3~4", "4~5", "5~6", "6+"]
        df2 = df.copy()
        df2["mag_range"] = pd.cut(df2["magnitude"], bins=bins, labels=labels)
        dist = df2["mag_range"].value_counts().sort_index()
        st.bar_chart(dist)

        st.subheader("🗂 최근 지진 목록 (최대 20건)")
        recent_df = df.sort_values("time", ascending=False).head(20)[["time", "magnitude", "depth", "place"]].copy()
        recent_df.columns = ["발생 시간", "규모", "깊이(km)", "위치"]
        recent_df["발생 시간"] = recent_df["발생 시간"].dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(recent_df, use_container_width=True)
