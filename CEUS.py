import cv2
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import imageio.v3 as iio
import os
import time

st.set_page_config(layout="wide")
st.title("Contrast-Enhanced Ultrasound ROI Analysis")

# --- Upload video ---
uploaded_file = st.file_uploader("Upload a video", type=["avi", "mp4", "mov"])
if uploaded_file is not None:
    # Preserve extension
    _, ext = os.path.splitext(uploaded_file.name)
    temp_filename = f"temp_video{ext}"

    with open(temp_filename, "wb") as f:
        f.write(uploaded_file.read())

    # --- Load video frames (force grayscale) ---
    frames = []
    try:
        for frame in iio.imiter(temp_filename):
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frames.append(gray)
    except Exception as e:
        st.error(f"❌ Could not read video. Error: {e}")
        st.stop()

    if len(frames) == 0:
        st.error("❌ No frames extracted. Please try another video.")
        st.stop()

    frames = np.stack(frames, axis=2)  # shape (H, W, N)
    st.success(f"Loaded {frames.shape[2]} frames, size {frames.shape[0]}x{frames.shape[1]}")

    # --- Basic Video Playback ---
    st.subheader("📽 Video Playback")
    play_col, slider_col = st.columns([1, 4])

    with play_col:
        play = st.button("▶ Play")

    with slider_col:
        frame_idx = st.slider("Frame", 0, frames.shape[2]-1, 0)

    # If play pressed: animate
    if play:
        placeholder = st.empty()
        for i in range(frame_idx, frames.shape[2]):
            img = Image.fromarray(frames[:, :, i])   # grayscale frame
            placeholder.image(img, caption=f"Frame {i}", use_container_width=True)
            time.sleep(0.05)  # playback speed
        st.stop()

    # Show selected frame
    source_frame = frames[:, :, frame_idx]
    st.image(source_frame, caption=f"Frame {frame_idx}", use_container_width=True)

    # Convert to RGB for ROI canvas background
    bg_img = Image.fromarray(cv2.cvtColor(source_frame, cv2.COLOR_GRAY2RGB))

    # --- ROI 1: Target ---
    st.subheader("Draw Target ROI (blue)")
    canvas_target = st_canvas(
        fill_color="rgba(0, 0, 255, 0.3)", 
        stroke_width=2,
        stroke_color="blue",
        background_image=bg_img,
        update_streamlit=True,
        height=source_frame.shape[0],
        width=source_frame.shape[1],
        drawing_mode="polygon",
        key="roi_target",
    )

    # --- ROI 2: Comparison ---
    st.subheader("Draw Comparison ROI (red)")
    canvas_compare = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)", 
        stroke_width=2,
        stroke_color="red",
        background_image=bg_img,
        update_streamlit=True,
        height=source_frame.shape[0],
        width=source_frame.shape[1],
        drawing_mode="polygon",
        key="roi_compare",
    )

    def polygon_to_mask(canvas_result, shape):
        """Convert drawn polygon to binary mask"""
        if canvas_result.json_data is None:
            return None
        if len(canvas_result.json_data["objects"]) == 0:
            return None
        polygon = canvas_result.json_data["objects"][0]["path"]
        pts = np.array([[p[1], p[2]] for p in polygon if p[0] == 'L'], dtype=np.int32)
        mask = np.zeros(shape, dtype=np.uint8)
        if len(pts) > 2:
            cv2.fillPoly(mask, [pts], 1)
            return mask
        return None

    mask_target = polygon_to_mask(canvas_target, source_frame.shape)
    mask_compare = polygon_to_mask(canvas_compare, source_frame.shape)

    if mask_target is not None and mask_compare is not None:
        st.success("Both ROIs defined. Calculating enhancement curves...")

        brightness_target = []
        brightness_compare = []
        for i in range(frames.shape[2]):
            frame = frames[:, :, i]
            vals_target = frame[mask_target > 0]
            vals_compare = frame[mask_compare > 0]
            brightness_target.append(np.mean(vals_target) if len(vals_target) > 0 else 0)
            brightness_compare.append(np.mean(vals_compare) if len(vals_compare) > 0 else 0)

        # --- Plot results ---
        fig, ax = plt.subplots()
        ax.plot(brightness_target, label="Target ROI", color="blue", linewidth=2)
        ax.plot(brightness_compare, label="Comparison ROI", color="red", linewidth=2)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Mean Intensity")
        ax.legend()
        st.pyplot(fig)

        # --- Prepare CSV for download ---
        df = pd.DataFrame({
            "Frame": np.arange(len(brightness_target)),
            "Target_ROI": brightness_target,
            "Comparison_ROI": brightness_compare
        })

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download enhancement curves as CSV",
            data=csv,
            file_name="roi_curves.csv",
            mime="text/csv",
        )
