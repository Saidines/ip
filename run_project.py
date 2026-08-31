import os
import cv2
import random
import glob
import shutil
from ultralytics import YOLO

BASE_DIR = os.getcwd()
DATASET_DIR = os.path.join(BASE_DIR, "vehicle_dataset")
RUNS_DIR = os.path.join(BASE_DIR, "runs")
VISUALS_DIR = os.path.join(BASE_DIR, "week1_visuals")
os.makedirs(RUNS_DIR, exist_ok=True)
os.makedirs(VISUALS_DIR, exist_ok=True)

print("--- STEP 1: Resetting and Organizing Dataset ---")
yolo_img_dir = os.path.join(DATASET_DIR, "train/images")
yolo_lbl_dir = os.path.join(DATASET_DIR, "train/labels")

# Delete old folders to start completely fresh
if os.path.exists(yolo_img_dir):
    shutil.rmtree(yolo_img_dir)
if os.path.exists(yolo_lbl_dir):
    shutil.rmtree(yolo_lbl_dir)
os.makedirs(yolo_img_dir, exist_ok=True)
os.makedirs(yolo_lbl_dir, exist_ok=True)

# Walk and find all images and labels in the original download
all_images = []
all_labels = []
for root, dirs, files in os.walk(DATASET_DIR):
    if "train" in root: continue  # Skip the folder we are creating
    for f in files:
        if f.endswith(('.jpg', '.png', '.jpeg')):
            all_images.append(os.path.join(root, f))
        elif f.endswith('.txt'):
            all_labels.append(os.path.join(root, f))

print(f"Found {len(all_images)} images and {len(all_labels)} labels in original folders.")

# Create a map of base filenames to label paths so we can match them
label_map = {}
for lbl_path in all_labels:
    base = os.path.splitext(os.path.basename(lbl_path))[0]
    label_map[base] = lbl_path

# Copy matched pairs into the train/ folder
copied_count = 0
for img_path in all_images:
    base = os.path.splitext(os.path.basename(img_path))[0]
    if base in label_map:
        shutil.copy(img_path, yolo_img_dir)
        shutil.copy(label_map[base], yolo_lbl_dir)
        copied_count += 1

print(f"Successfully copied {copied_count} image-label pairs to train/images and train/labels.")

if copied_count == 0:
    print("CRITICAL ERROR: Could not match any images to labels. Check your folder structure.")
    exit()

print("\n--- STEP 2: Converting string classes to YOLO integer IDs ---")
class_mapping = {}
current_id = 0

# Convert words like 'Car' to numbers like '0'
for lbl_file in os.listdir(yolo_lbl_dir):
    lbl_path = os.path.join(yolo_lbl_dir, lbl_file)
    with open(lbl_path, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5: continue
        
        class_name = parts[0]
        if not class_name.isdigit():
            if class_name not in class_mapping:
                class_mapping[class_name] = current_id
                current_id += 1
            class_id = class_mapping[class_name]
        else:
            class_id = int(class_name)
            
        new_line = f"{class_id} {' '.join(parts[1:])}"
        new_lines.append(new_line)
        
    with open(lbl_path, 'w') as f:
        f.write("\n".join(new_lines))

print(f"Class mapping: {class_mapping}")

print("\n--- STEP 3: Creating data.yaml ---")
max_classes = max(len(class_mapping), 1)
if class_mapping:
    names_dict = {v: k for k, v in class_mapping.items()}
else:
    names_dict = {0: 'object'}

yaml_content = f"""
path: {DATASET_DIR}
train: train/images  
val: train/images  

nc: {max_classes}
names: {names_dict}
"""
yaml_path = os.path.join(DATASET_DIR, "baseline_data.yaml")
with open(yaml_path, 'w') as f:
    f.write(yaml_content)
print("data.yaml created.")

print("\n--- STEP 4: Visualising Bounding Boxes ---")
image_files = [f for f in os.listdir(yolo_img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
sample_images = random.sample(image_files, min(3, len(image_files)))

for img_file in sample_images:
    img_path = os.path.join(yolo_img_dir, img_file)
    lbl_path = os.path.join(yolo_lbl_dir, img_file.rsplit('.', 1)[0] + '.txt')
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    
    if os.path.exists(lbl_path):
        with open(lbl_path, 'r') as f: lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5: continue
            class_id = int(parts[0])
            x_c, y_c = float(parts[1]) * w, float(parts[2]) * h
            box_w, box_h = float(parts[3]) * w, float(parts[4]) * h
            x1 = int(x_c - box_w / 2)
            y1 = int(y_c - box_h / 2)
            x2 = int(x_c + box_w / 2)
            y2 = int(y_c + box_h / 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, str(class_id), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(VISUALS_DIR, img_file), img)
print(f"Saved {len(sample_images)} samples to {VISUALS_DIR}")

print("\n--- STEP 5: Training Baseline Models ---")
import torch
print(f"GPU Available: {torch.cuda.is_available()}")

models_to_train = ['yolov8n.pt', 'yolo11n.pt'] 

for model_name in models_to_train:
    print(f"\nTraining Baseline for {model_name}...")
    model = YOLO(model_name)
    model.train(
        data=yaml_path,
        epochs=50,
        imgsz=640,
        batch=16,
        degrees=0,
        flipud=0,
        project=RUNS_DIR,
        name=f"baseline_{model_name}"
    )
    print(f"Finished training {model_name}")

print("\n--- STEP 6: Consolidating Baseline Metrics ---")
import pandas as pd
result_files = glob.glob(os.path.join(RUNS_DIR, "baseline_*/results.csv"))
data_matrix = []
for file_path in result_files:
    run_name = os.path.basename(os.path.dirname(file_path))
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    last_row = df.iloc[-1]
    try:
        time_hours = df['train_time'].iloc[-1] / 3600 if 'train_time' in df else df['time_total_hours'].iloc[-1]
    except:
        time_hours = "N/A"
    data_matrix.append({
        "Run Name": run_name,
        "Precision": last_row.get('metrics/precision(B)', None),
        "Recall": last_row.get('metrics/recall(B)', None),
        "mAP@50": last_row.get('metrics/mAP50(B)', None),
        "mAP@50-95": last_row.get('metrics/mAP50-95(B)', None),
        "Time (hrs)": time_hours
    })
results_df = pd.DataFrame(data_matrix)
print("\n--- BASELINE METRICS MATRIX ---")
print(results_df.to_string(index=False))
results_df.to_csv(os.path.join(BASE_DIR, 'baseline_metrics.csv'), index=False)
print("\nDone! Check 'runs' and 'week1_visuals' folders.")
