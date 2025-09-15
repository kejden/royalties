import streamlit as st
import os
import io
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from moviepy.editor import VideoFileClip
from streamlit_image_coordinates import streamlit_image_coordinates

VIDEO_FILE_PATH = "../movies/Kiler.mp4"

st.set_page_config(layout="wide", page_title="Movie Analysis Dashboard")

@st.cache_resource
def get_video_duration(video_path):
    """Wczytuje plik wideo i zwraca jego czas trwania w sekundach."""
    if not os.path.exists(video_path):
        return None
    with VideoFileClip(video_path) as clip:
        return int(clip.duration)

TOTAL_VIDEO_DURATION_SECONDS = get_video_duration(VIDEO_FILE_PATH)

if TOTAL_VIDEO_DURATION_SECONDS is None:
    st.error(f"Plik wideo nie został znaleziony w: {VIDEO_FILE_PATH}")
    st.info("Zaktualizuj zmienną `VIDEO_FILE_PATH` w skrypcie.")
    st.stop()

if 'global_start_time' not in st.session_state:
    st.session_state.global_start_time = 0

ACTORS_DATA = {
    "Cezary Pazura": {
        "thumbnail": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRntQJjsN_F2wq9PnGo8N5fIBu2rKic9dEgUg&s",
        "appearance_percentage": 35.6,
        "timestamps": [(3212, 3240), (3900,4000), (4200, 4250), (4510,4560)]
    },
    "Janusz Rewinski": {
        "thumbnail": "https://upload.wikimedia.org/wikipedia/commons/d/dc/Rewinski_Janusz.jpg",
        "appearance_percentage": 18.2,
        "timestamps": [(1898, 1920), (2001, 2090), (2120, 2200)]
    },
    "Zona": {
        "thumbnail": "https://encrypted-tbn1.gstatic.com/images?q=tbn:ANd9GcRvfxcINpTJE5VZymMLQqzT_pjY61DiTk2mhyRJyRsBJppxn707ssTO8orXzGDa5ChQxkzzLpNjPriVCyye4F59HZEDWxzJ18KSvfPKXik",
        "appearance_percentage": 62.9,
        "timestamps": [(2048, 2200), (3000,3050), (4500, 4600)]
    }
}

max_timestamp_from_data = 0
for actor in ACTORS_DATA.values():
    for start, end in actor.get("timestamps", []):
        if end > max_timestamp_from_data:
            max_timestamp_from_data = end

TIMELINE_DURATION = max(TOTAL_VIDEO_DURATION_SECONDS, max_timestamp_from_data)

def format_time(seconds):
    if seconds is None: return "00:00:00"
    hours = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{mins:02d}:{secs:02d}"

@st.cache_resource
def create_timeline_plot(timestamps, total_duration, current_video_time=0):
    fig, ax = plt.subplots(figsize=(20, 0.2), dpi=100)
    fig.patch.set_facecolor('#F0F2F6')
    ax.set_facecolor('#F0F2F6')
    ax.add_patch(patches.Rectangle((0, 0), total_duration, 1, facecolor='#d3d3d3', edgecolor='none'))
    for start, end in timestamps:
        if start < total_duration:
            ax.add_patch(patches.Rectangle((start, 0), min(end, total_duration) - start, 1, facecolor='royalblue', edgecolor='none'))
    ax.axvline(x=current_video_time, color='red', linestyle='-', linewidth=2)
    ax.set_xlim(0, total_duration)
    ax.set_ylim(0, 1)
    ax.axis('off')
    return fig

def play_previous_segment(actor_timestamps, current_time):
    sorted_timestamps = sorted(actor_timestamps, key=lambda x: x[0], reverse=True)
    prev_segment_start = current_time
    for start, end in sorted_timestamps:
        if start < current_time:
            prev_segment_start = start
            break
    st.session_state.global_start_time = prev_segment_start

def play_next_segment(actor_timestamps, current_time):
    sorted_timestamps = sorted(actor_timestamps, key=lambda x: x[0])
    next_segment_start = current_time
    for start, end in sorted_timestamps:
        if start > current_time:
            next_segment_start = start
            break
    st.session_state.global_start_time = next_segment_start

left_col, right_col = st.columns([2, 1])

with left_col:
    st.header("Odtwarzacz wideo")
    st.video(VIDEO_FILE_PATH, start_time=st.session_state.global_start_time)

with right_col:
    st.header("Aktorzy")
    actor_list = list(ACTORS_DATA.keys())
    selected_actor_id = st.selectbox("Wybierz aktora z listy", actor_list)
    actor_info = ACTORS_DATA.get(selected_actor_id, {})

    if actor_info:
        thumb_col, detail_col = st.columns([1, 2])
        with thumb_col:
            st.image(actor_info["thumbnail"], width=120)
        with detail_col:
            st.subheader(selected_actor_id)
            st.metric(label="Pojawia się w", value=f"{actor_info['appearance_percentage']}% wideo")
    else:
        st.warning("Nie znaleziono danych dla wybranej osoby.")

st.divider()

if actor_info:
    timeline_fig = create_timeline_plot(
        actor_info.get("timestamps", []),
        TIMELINE_DURATION,
        current_video_time=st.session_state.global_start_time
    )
    buf = io.BytesIO()
    timeline_fig.savefig(buf, format="png", bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    image_for_coords = Image.open(buf)

    comp_key = f"timeline_click_{selected_actor_id}"
    value = streamlit_image_coordinates(
        image_for_coords,
        key=comp_key
    )

    last_click_key = f"last_processed_timeline_click_{selected_actor_id}"
    if last_click_key not in st.session_state:
        st.session_state[last_click_key] = None

    if value:
        clicked_x = value.get('x')
        clicked_y = value.get('y')
        image_width = image_for_coords.width
        if image_width and clicked_x is not None:
            new_start_time = int((clicked_x / image_width) * TIMELINE_DURATION)
            last = st.session_state.get(last_click_key)
            if last is None or last.get('x') != clicked_x or last.get('y') != clicked_y:
                st.session_state.global_start_time = new_start_time
                st.session_state[last_click_key] = {'x': clicked_x, 'y': clicked_y, 'time': new_start_time}
                st.rerun()

    st.write("")
    col1, col2, col3 = st.columns([1, 4, 1])
    actor_timestamps = actor_info.get("timestamps", [])
    current_time = st.session_state.global_start_time
    
    is_first_scene = not any(start < current_time for start, end in actor_timestamps)
    is_last_scene = not any(start > current_time for start, end in actor_timestamps)

    with col1:
        st.button(
            "|◁ Poprzednia scena", 
            on_click=play_previous_segment, 
            args=(actor_timestamps, current_time),
            disabled=is_first_scene
        )
    with col3:
        st.button(
            "Następna scena ▷|", 
            on_click=play_next_segment, 
            args=(actor_timestamps, current_time),
            disabled=is_last_scene
        )
else:
    st.info("Wybierz osobę z listy, aby zobaczyć i używać osi czasu.")
