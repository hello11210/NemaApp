import streamlit as st
import cv2
import torch
import numpy as np
import tempfile
import os
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import tempfile

# For data pushing
import pandas as pd
import os

# Download model from external link
import requests

# To download csv files
import io

st.set_page_config(page_title="Nematode Detector", layout="centered")
st.title("Nematode Detector with YOLOv8")
st.sidebar.header("Detection Settings")
class_labels = {0: "Nematode", 1: "Debris"}
selected_classes = [
    class_id for class_id, label in class_labels.items()
    if st.sidebar.checkbox(label, value=(class_id == 0))
]
confidence_thresh = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)


MODEL_URL = "https://huggingface.co/dman3/classifier/resolve/main/cweights.pt"
MODEL_PATH = "models/model.pt"

def download_model():
    os.makedirs("models", exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model weights..."):
            r = requests.get(MODEL_URL, stream=True)
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        st.success("Model downloaded successfully!")
# Before loading model
download_model()
# Then load your model like this
classifier = YOLO(MODEL_PATH)

MODEL_URL2 = "https://huggingface.co/dman3/detector/resolve/main/last.pt"
MODEL_PATH2 = "models/model2.pt"

def download_model():
    os.makedirs("models", exist_ok=True)
    if not os.path.exists(MODEL_PATH2):
        with st.spinner("Downloading model weights..."):
            r = requests.get(MODEL_URL2, stream=True)
            with open(MODEL_PATH2, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        st.success("Model downloaded successfully!")
# Before loading model
download_model()
# Then load your model like this
model = YOLO(MODEL_PATH2)




#model = YOLO("models/last.pt")
#classifier = YOLO("models/cweights.pt")
#print(classifier.task)

if "video_tracking_data" not in st.session_state:
    st.session_state.video_tracking_data = []




#________________________________________________________________________________________________________________________________________
def detect_on_video(video_file):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    tfile.flush()

    cap = cv2.VideoCapture(tfile.name)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    output_path = os.path.join(tempfile.gettempdir(), "processed_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)

    frame_idx = 0
    tracking_data = []
    unique_ids = set()
    id_frame_count = {}

    st.session_state.video_classification = []
    st.session_state.video_classification_data = []  # ✅  

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True)[0]
        filtered_boxes = []

        for box in results.boxes:
            track_id = int(box.id[0]) if box.id is not None else None
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if cls in selected_classes and conf > confidence_thresh:
                filtered_boxes.append(box)
                if track_id is not None:
                    unique_ids.add(track_id)
                    id_frame_count[track_id] = id_frame_count.get(track_id, 0) + 1
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = xyxy
                    tracking_data.append({
                        "id": track_id,
                        "cls": cls,  # Save the class (nematode, debris, etc.)
                        "frame": frame_idx,
                        "bbox": [x1, y1, x2, y2],
                        "crop": frame[y1:y2, x1:x2]
                    })

        results.boxes = filtered_boxes
        annotated_frame = results.plot()
        out.write(annotated_frame)

        frame_idx += 1
        progress_bar.progress(min(frame_idx / frame_count, 1.0))

    cap.release()
    out.release()
    progress_bar.empty()

    st.session_state.video_processed = output_path

    # === Classification per ID (if seen in ≥100 frames) ===
    classified_ids = set()
    st.session_state.video_classification_data = []

    for track in tracking_data:
        tid = track["id"]
        cls = track["cls"]

        # ✅ Skip if already classified or not a nematode (class 0)
        if tid in classified_ids or cls != 0:
            continue

        # Skip if seen in fewer than 100 frames
        if id_frame_count.get(tid, 0) < 100:
            continue

        classified_ids.add(tid)
        crop = track["crop"]

        if crop.size == 0:
            continue

        cls_result = classifier.predict(crop, imgsz=224, conf=0.25, verbose=False)[0]
        # === After successful classification of each valid nematode ID ===
        if hasattr(cls_result, "boxes") and cls_result.boxes and len(cls_result.boxes.cls) > 0:
            best_idx = int(torch.argmax(cls_result.boxes.conf))
            class_id = int(cls_result.boxes.cls[best_idx])
            confidence = float(cls_result.boxes.conf[best_idx])
            class_name = classifier.names[class_id]

            st.session_state.video_classification.append(
                f"Nematode ID {tid}: `{class_name}` ({confidence:.2f})"
            )

            # ✅ Add a new formatted row per valid nematode
            row = {
                "filename": video_file.name,
                "id": tid  # ✅ Use lowercase 'id' to match what's used later
            }
            for species in classifier.names.values():
                row[species] = f"{confidence:.2f}" if species == class_name else ""
            st.session_state.video_classification_data.append(row)
        else:
            st.session_state.video_classification.append(
                f"Nematode ID {tid}: Could not classify."
            )
            
    #st.success(f"Video processing complete - Unique tracked objects: {len(unique_ids)}")
    #print("\n" + str(video_classification_data) + "\n\n")
    st.success(f"Video processing complete")



#________________________________________________________________________________________________________________________________________
def detect_on_image(uploaded_image, show_output=True):
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    results = model(img)[0]

    filtered_boxes = []
    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        if cls in selected_classes and conf > confidence_thresh:
            filtered_boxes.append(box)
    results.boxes = filtered_boxes
    st.session_state.filtered_boxes = filtered_boxes  # for classification
    st.session_state.original_image = img             # original unannotated image  
    # ======================================
    if show_output:
        annotated = results.plot()
        return annotated

    return img



#________________________________________________________________________________________________________________________________________
def push_to_csv(classification_data, species_list):
    print("\nInside push to csv ________")
    print(classification_data)
    st.markdown("### Push Classifications to CSV Database")

    if "csv_push_mode" not in st.session_state:
        st.session_state.csv_push_mode = None

    col1, col2 = st.columns(2)

    with col1:
        if st.button("New CSV File", key="btn_new_csv_image"):
            st.session_state.csv_push_mode = "new"

    with col2:
        if st.button("Append to Existing CSV", key="btn_append_csv_image"):
            st.session_state.csv_push_mode = "append"
            
    def convert_to_csv_df(data):
        rows = []

        for item in data:
            # Initialize the row with empty species columns
            row = {species: "" for species in species_list}
            row["filename"] = item.get("filename", "")
            row["ID"] = item.get("id", "")

            # Find the species in the item dictionary
            for key in item:
                if key in species_list:
                    row[key] = item[key]

            rows.append(row)

        
        return pd.DataFrame(rows)

    # === New CSV File ===
    if st.session_state.csv_push_mode == "new":
        new_name = st.text_input("Enter name for new CSV file (e.g., `my_data.csv`)", key="new_csv_name")
        if new_name and st.button("Save to New CSV"):
            df = convert_to_csv_df(classification_data)
            # Reorder columns: filename, ID, species...
            df = df[["filename", "ID"] + species_list]
            df.to_csv(new_name, index=False)
            st.rerun()
            st.success(f"Saved classification results to `{new_name}`.")
            st.session_state.csv_push_mode = None

    # === Append to Existing CSV ===
    elif st.session_state.csv_push_mode == "append":
        existing_name = st.text_input("Enter name of existing CSV file to append to", key="existing_csv_name")
        if existing_name and st.button("Append to Existing CSV"):
            new_df = convert_to_csv_df(classification_data)
            expected_columns = ["filename", "ID"] + species_list
            existing_columns = [col for col in expected_columns if col in new_df.columns]
            new_df = new_df[existing_columns]
            if os.path.exists(existing_name):
                try:
                    old_df = pd.read_csv(existing_name)
                    combined_df = pd.concat([old_df, new_df], ignore_index=True)
                    combined_df.to_csv(existing_name, index=False)
                    st.rerun()
                    st.success(f"Appended classification results to `{existing_name}`.")
                except Exception as e:
                    st.error(f"⚠️ Error reading CSV: {e}")
            else:
                st.warning(f"File `{existing_name}` not found. Please try again.")
            st.session_state.csv_push_mode = None


#________________________________________________________________________________________________________________________________________
def push_multiple_images_to_csv(classification_data, species_list):
    st.markdown("### Push Classifications to CSV Database")

    if "csv_push_mode_multi" not in st.session_state:
        st.session_state.csv_push_mode_multi = None

    col1, col2 = st.columns(2)

    with col1:
        if st.button("New CSV File", key="btn_new_csv_multi"):
            st.session_state.csv_push_mode_multi = "new"

    with col2:
        if st.button("Append to Existing CSV", key="btn_append_csv_multi"):
            st.session_state.csv_push_mode_multi = "append"

    def convert_to_csv_df(data):
        rows = []

        for item in data:
            row = {species: "" for species in species_list}
            row["filename"] = item.get("filename", "")
            row["ID"] = item.get("id", "")

            for key in item:
                if key in species_list:
                    row[key] = item[key]

            rows.append(row)

        return pd.DataFrame(rows)

    if st.session_state.csv_push_mode_multi == "new":
        new_name = st.text_input("Enter name for new CSV file (e.g., `my_data.csv`)", key="new_csv_name_multi")
        if new_name and st.button("Save to New CSV", key="btn_save_new_multi"):
            df = convert_to_csv_df(classification_data)
            df = df[["filename", "ID"] + species_list]
            df.to_csv(new_name, index=False)
            st.rerun()
            st.success(f"Saved classification results to `{new_name}`.")
            st.session_state.csv_push_mode_multi = None

    elif st.session_state.csv_push_mode_multi == "append":
        existing_name = st.text_input("Enter name of existing CSV file to append to", key="existing_csv_name_multi")
        if existing_name and st.button("Append to Existing CSV", key="btn_save_append_multi"):
            new_df = convert_to_csv_df(classification_data)
            expected_columns = ["filename", "ID"] + species_list
            existing_columns = [col for col in expected_columns if col in new_df.columns]
            new_df = new_df[existing_columns]

            if os.path.exists(existing_name):
                try:
                    old_df = pd.read_csv(existing_name)
                    combined_df = pd.concat([old_df, new_df], ignore_index=True)
                    combined_df.to_csv(existing_name, index=False)
                    st.rerun()
                    st.success(f"Appended classification results to `{existing_name}`.")
                except Exception as e:
                    st.error(f"⚠️ Error reading CSV: {e}")
            else:
                st.warning(f"File `{existing_name}` not found. Please try again.")
            st.session_state.csv_push_mode_multi = None



#________________________________________________________________________________________________________________________________________
import uuid

def push_video_to_csv(classification_data, species_list, widget_suffix=None):
    if not widget_suffix:
        widget_suffix = str(uuid.uuid4())  # unique ID per call

    st.markdown("### Push Video Classifications to CSV Database")

    key_prefix = f"video_csv_{widget_suffix}"
    if f"{key_prefix}_mode" not in st.session_state:
        st.session_state[f"{key_prefix}_mode"] = None

    col1, col2 = st.columns(2)

    with col1:
        if st.button("New CSV File", key=f"{key_prefix}_new"):
            st.session_state[f"{key_prefix}_mode"] = "new"

    with col2:
        if st.button("Append to Existing CSV", key=f"{key_prefix}_append"):
            st.session_state[f"{key_prefix}_mode"] = "append"

    def convert_video_classification_to_df(data, species_list):
        rows = []
        for item in data:
            row = {species: "" for species in species_list}
            row["filename"] = item.get("filename", "")
            row["ID"] = item.get("id", "")
            for species in species_list:
                if species in item:
                    row[species] = item[species]
            rows.append(row)
        return pd.DataFrame(rows)

    mode = st.session_state[f"{key_prefix}_mode"]

    if mode == "new":
        new_name = st.text_input("Enter name for new CSV file (e.g., `video_data.csv`)", key=f"{key_prefix}_new_input")
        if new_name and st.button("Save to New CSV", key=f"{key_prefix}_save_new"):
            df = convert_video_classification_to_df(classification_data, species_list)
            try:
                df = df[["filename", "ID"] + species_list]
                df.to_csv(new_name, index=False)
                st.rerun()
                st.success(f"Saved video classification results to `{new_name}`.")
                st.session_state[f"{key_prefix}_mode"] = None
            except Exception as e:
                st.error(f"Failed to save: {e}")

    elif mode == "append":
        existing_name = st.text_input("Enter existing CSV file name", key=f"{key_prefix}_append_input")
        if existing_name and st.button("Append to Existing CSV", key=f"{key_prefix}_append_btn"):
            new_df = convert_video_classification_to_df(classification_data, species_list)
            try:
                new_df = new_df[["filename", "ID"] + species_list]
                if os.path.exists(existing_name):
                    old_df = pd.read_csv(existing_name)
                    combined_df = pd.concat([old_df, new_df], ignore_index=True)
                    combined_df.to_csv(existing_name, index=False)
                    st.rerun()
                    st.success(f"Appended video classification results to `{existing_name}`.")
                else:
                    st.warning(f"File `{existing_name}` not found.")
                st.session_state[f"{key_prefix}_mode"] = None
            except Exception as e:
                st.error(f"Error during append: {e}")
    

#________________________________________________________________________________________________________________________________________
# Webcam transformer class
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = model(img)[0]
        annotated = results.plot()
        return annotated

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = model.track(frame, persist=True)[0]

        filtered_boxes = []
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if cls == 0 and conf > 0.5:
                filtered_boxes.append(box)
        results.boxes = filtered_boxes

        annotated = results.plot()
        return annotated









with st.sidebar:
    st.markdown("### Download Existing CSVs")

    csv_files = [f for f in os.listdir() if f.endswith(".csv")]
    
    if not csv_files:
        st.info("No CSV files found in directory.")
    else:
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label=f"{csv_file}",
                    data=csv_buffer.getvalue(),
                    file_name=csv_file,
                    mime="text/csv",
                    key=f"download_{csv_file}"
                )
            except Exception as e:
                st.warning(f"⚠️ Could not load {csv_file}: {e}")











#________________________________________________________________________________________________________________________________________
option = st.radio("Select Input Type", ["Image", "Multiple Images", "Video", "Live Camera"])

if option == "Image":
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        annotated = detect_on_image(uploaded_image)
        st.image([annotated], caption=["Detected Image"], channels="BGR", use_container_width=True)
        is_success, image_bytes = cv2.imencode(".jpg", annotated)
        if is_success:
            st.download_button("Download Processed Image", image_bytes.tobytes(), file_name="processed_image.jpg", mime="image/jpeg")
            if st.button("Run Classification on Detected Nematodes"):
                original_img = st.session_state.original_image.copy()
                classified = False


                classification_results = []
                image_filename = uploaded_image.name if uploaded_image else "unknown_image.jpg"
                # List all known species (update as needed)
                species_list = [
                    'grass', 'stunt', 'leaf', 'predatory soil', 'axonchium', 'dorylaimid', 'spiral', 'needle', 'root-knot', 'ring', 'miconchus', 'free-living', 'lesion', 'pristionchus', 'reniform', 'stem'
                ]
                species_list = [s.strip().lower() for s in species_list]


                st.subheader("Classification Results")

                for i, box in enumerate(st.session_state.get("filtered_boxes", [])):
                    cls = int(box.cls[0])
                    if cls != 0:  # Only classify nematodes (class 0)
                        continue

                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = xyxy

                    # Clamp coordinates
                    h, w = original_img.shape[:2]
                    x1 = max(0, min(x1, w - 1))
                    x2 = max(x1 + 1, min(x2, w))
                    y1 = max(0, min(y1, h - 1))
                    y2 = max(y1 + 1, min(y2, h))

                    crop_bgr = original_img[y1:y2, x1:x2]

                    crop_path = f"crop_{i}.jpg"
                    cv2.imwrite(crop_path, crop_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

                    cls_results = classifier.predict(
                        source=crop_path,
                        conf=0.25,
                        iou=0.7,
                        imgsz=640,
                        device='cpu',
                        save=False,
                        half=False,
                        stream=False,
                        verbose=False
                    )

                    cls_result = cls_results[0]

                    with st.container():
                        cols = st.columns([1, 2])
                        with cols[0]:
                            st.image(crop_bgr, caption=f"Nematode {i+1}", channels="BGR", use_container_width=True)

                        with cols[1]:
                            if cls_result.boxes and len(cls_result.boxes.cls) > 0:
                                best_idx = int(torch.argmax(cls_result.boxes.conf))
                                class_id = int(cls_result.boxes.cls[best_idx])
                                confidence = float(cls_result.boxes.conf[best_idx])
                                class_name = classifier.names[class_id]

                                st.markdown(f"**Species:** `{class_name}`")
                                st.markdown(f"**Confidence:** `{confidence*100:.2f}%`")
                                st.progress(confidence)

                                # Normalize class name
                                class_name = classifier.names[class_id].strip().lower()

                                # Build row for this nematode
                                row = {
                                    "filename": uploaded_image.name,
                                    "id": i + 1
                                }

                                # Fill in only the correct species column with confidence
                                #print("\n")
                                #print(class_name)
                                #print("\n\n\n\n")
                                for species in species_list:
                                    print(species) 
                                    if species == class_name:
                                        print("\nSUCCESSSSS\n")
                                        row[species] = f"{confidence:.2f}"
                                    # else:
                                    #     row[species] = ""

                                classification_results.append(row)
                                print("\n\n")
                                print(classification_results)
                                classified = True
                            else:
                                st.warning(f"Nematode {i+1} could not be classified.")

                if not classified:
                    st.warning("No valid nematode crops were classified.")
                else:
                    st.session_state.image_classification_results = classification_results
                    st.session_state.species_list = species_list  # for CSV writing
            if "image_classification_results" in st.session_state:
                push_to_csv(
                    classification_data=st.session_state.image_classification_results,
                    species_list=st.session_state.species_list
                )
                
elif option == "Multiple Images":
    uploaded_images = st.file_uploader("Upload multiple images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_images:
        species_list = [
            'grass', 'stunt', 'leaf', 'predatory soil', 'axonchium', 'dorylaimid',
            'spiral', 'needle', 'root-knot', 'ring', 'miconchus', 'free-living',
            'lesion', 'pristionchus', 'reniform', 'stem'
        ]
        species_list = [s.strip().lower() for s in species_list]

        # Step 1: Prompt for CSV write mode (but don't trigger push yet)
        if "multi_prompt_csv_push_mode" not in st.session_state:
            st.session_state.multi_prompt_csv_push_mode = None
        push_mode_col1, push_mode_col2 = st.columns(2)
        with push_mode_col1:
            if st.button("New CSV File", key="btn_new_csv_multi_prompt"):
                st.session_state.multi_prompt_csv_push_mode = "new"
        with push_mode_col2:
            if st.button("Append to Existing CSV", key="btn_append_csv_multi_prompt"):
                st.session_state.multi_prompt_csv_push_mode = "append"

        # Step 2: Input field based on user selection
        mode = st.session_state.multi_prompt_csv_push_mode
        name_entered = None
        if mode == "new":
            name_entered = st.text_input("Enter name for new CSV file", key="new_csv_name_multi_prompt")
        elif mode == "append":
            name_entered = st.text_input("Enter name of existing CSV file to append to", key="existing_csv_name_multi_prompt")

        # Step 3: Once filename is provided, run detection/classification/push
        if name_entered:
            all_results = []
            progress_bar = st.progress(0)

            for idx, uploaded_image in enumerate(uploaded_images):
                image = detect_on_image(uploaded_image, show_output=False)
                original_img = st.session_state.original_image.copy()
                image_results = []

                for i, box in enumerate(st.session_state.get("filtered_boxes", [])):
                    cls = int(box.cls[0])
                    if cls != 0:
                        continue

                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = map(int, xyxy)
                    h, w = original_img.shape[:2]
                    x1 = max(0, min(x1, w - 1))
                    x2 = max(x1 + 1, min(x2, w))
                    y1 = max(0, min(y1, h - 1))
                    y2 = max(y1 + 1, min(y2, h))

                    crop_bgr = original_img[y1:y2, x1:x2]
                    crop_path = f"crop_{uploaded_image.name}_{i}.jpg"
                    cv2.imwrite(crop_path, crop_bgr)

                    cls_results = classifier.predict(
                        source=crop_path, conf=0.25, iou=0.7, imgsz=640,
                        device='cpu', save=False, half=False, stream=False, verbose=False
                    )[0]

                    if cls_results.boxes and len(cls_results.boxes.cls) > 0:
                        best_idx = int(torch.argmax(cls_results.boxes.conf))
                        class_id = int(cls_results.boxes.cls[best_idx])
                        confidence = float(cls_results.boxes.conf[best_idx])
                        class_name = classifier.names[class_id].strip().lower()

                        row = {
                            "filename": uploaded_image.name,
                            "id": i + 1
                        }
                        for species in species_list:
                            row[species] = f"{confidence:.2f}" if species == class_name else ""
                        image_results.append(row)

                all_results.extend(image_results)
                progress_bar.progress((idx + 1) / len(uploaded_images))

            progress_bar.empty()

            # Step 4: Save to CSV
            def convert_to_csv_df(data):
                rows = []
                for item in data:
                    row = {species: "" for species in species_list}
                    row["filename"] = item.get("filename", "")
                    row["ID"] = item.get("id", "")
                    for key in item:
                        if key in species_list:
                            row[key] = item[key]
                    rows.append(row)
                return pd.DataFrame(rows)

            df = convert_to_csv_df(all_results)
            df = df[["filename", "ID"] + species_list]

            if mode == "new":
                df.to_csv(name_entered, index=False)
                st.success(f"Saved classification results to `{name_entered}`.")
            elif mode == "append":
                if os.path.exists(name_entered):
                    try:
                        old_df = pd.read_csv(name_entered)
                        combined_df = pd.concat([old_df, df], ignore_index=True)
                        combined_df.to_csv(name_entered, index=False)
                        st.success(f"Appended classification results to `{name_entered}`.")
                    except Exception as e:
                        st.error(f"⚠️ Error reading CSV: {e}")
                else:
                    st.warning(f"File `{name_entered}` not found. Please try again.")

            # Reset session state after push
            st.session_state.multi_prompt_csv_push_mode = None


elif option == "Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
    
    if uploaded_video:
        # Only process if the video has changed
        if (
            "last_uploaded_video_name" not in st.session_state
            or st.session_state.last_uploaded_video_name != uploaded_video.name
        ):
            detect_on_video(uploaded_video)
            st.session_state.last_uploaded_video_name = uploaded_video.name

        if "video_processed" in st.session_state:
            with open(st.session_state.video_processed, "rb") as f:
                st.download_button("Download Processed Video", f, file_name="processed_output.mp4", key="download_video_button")

        if st.session_state.get("video_classification"):
            st.markdown("### Final Classification Results per Nematode")
            for entry in st.session_state.video_classification:
                if isinstance(entry, str):
                    st.markdown(f"- {str(entry)}")
                elif isinstance(entry, dict):
                    label = entry.get("label") or entry.get("species") or "Unknown"
                    tid = entry.get("id", "N/A")
                    fname = entry.get("filename", "")
                    st.markdown(f"- {fname} | ID {tid}: `{label}`")
                else:
                    st.markdown(f"- {entry}")

            if st.session_state.get("video_classification_data"):
                species_list = list(classifier.names.values())
                push_video_to_csv(st.session_state.video_classification_data, species_list, widget_suffix="video")
        else:
            st.info("No classification results to show.")
            
            

elif option == "Live Camera":
    webrtc_streamer(key="live", video_transformer_factory=VideoTransformer)

