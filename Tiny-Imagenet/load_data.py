import os
import shutil

root = "..."
val_dir = os.path.join(root, "val")
img_dir = os.path.join(val_dir, "images")
ann_file = os.path.join(val_dir, "val_annotations.txt")

with open(ann_file, "r") as f:
    for line in f:
        img_name, class_id = line.strip().split("\t")[:2]

        class_dir = os.path.join(val_dir, class_id)
        os.makedirs(class_dir, exist_ok=True)

        src = os.path.join(img_dir, img_name)
        dst = os.path.join(class_dir, img_name)

        if os.path.exists(src):
            shutil.move(src, dst)

print("Done.")