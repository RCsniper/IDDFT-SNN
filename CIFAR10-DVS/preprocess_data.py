import os
import math
import numpy as np
from tqdm import tqdm

CLASS_NAMES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
)


def load_aedat_cifar10dvs(file_name):
    def skip_header(fp):
        pos = 0
        line = fp.readline()
        while line:
            try:
                s = line.decode().strip()
            except UnicodeDecodeError:
                break
            if len(s) == 0 or s[0] != '#':
                break
            pos += len(line)
            line = fp.readline()
        return pos

    with open(file_name, "rb") as fp:
        p = skip_header(fp)
        fp.seek(p)
        raw = np.frombuffer(fp.read(), dtype=">u4")

    raw_addr = raw[::2]
    t = raw[1::2]

    x = (raw_addr & 0xFE) >> 1
    y = (raw_addr & 0x7F00) >> 8
    p = (raw_addr & 1).astype(np.int64)

    x_new = 127 - y
    y_new = 127 - x
    p_new = 1 - p

    return {
        "t": t.astype(np.int64),
        "x": x_new.astype(np.int64),
        "y": y_new.astype(np.int64),
        "p": p_new.astype(np.int64),
    }


def integrate_events_to_frames_number(events, frames_number=10, H=128, W=128):
    t = events["t"]
    x = events["x"]
    y = events["y"]
    p = events["p"]

    n = len(t)
    frames = np.zeros((frames_number, 2, H, W), dtype=np.float32)

    if n == 0:
        return frames

    idx = np.linspace(0, n, frames_number + 1, dtype=np.int64)

    for i in range(frames_number):
        s, e = idx[i], idx[i + 1]
        if s >= e:
            continue
        xi = x[s:e]
        yi = y[s:e]
        pi = p[s:e]
        np.add.at(frames[i], (pi, yi, xi), 1)

    return frames


def preprocess_and_save(
    raw_root,
    save_root,
    frames_number=10,
    overwrite=False,
):

    os.makedirs(save_root, exist_ok=True)

    total_files = 0
    for class_name in CLASS_NAMES:
        src_dir = os.path.join(raw_root, "extract", class_name)
        if not os.path.isdir(src_dir):
            raise FileNotFoundError(f"Missing class dir: {src_dir}")

        dst_dir = os.path.join(save_root, class_name)
        os.makedirs(dst_dir, exist_ok=True)

        files = sorted([f for f in os.listdir(src_dir) if f.endswith(".aedat")])
        total_files += len(files)

        for fname in tqdm(files, desc=f"processing {class_name}"):
            src_path = os.path.join(src_dir, fname)
            dst_path = os.path.join(dst_dir, fname.replace(".aedat", f"_T{frames_number}.npz"))

            if (not overwrite) and os.path.exists(dst_path):
                continue

            events = load_aedat_cifar10dvs(src_path)
            frames = integrate_events_to_frames_number(
                events,
                frames_number=frames_number,
                H=128,
                W=128,
            )

            np.savez_compressed(dst_path, frames=frames.astype(np.float32))

    print(f"Done. Total files found: {total_files}")
    print(f"Saved to: {save_root}")


if __name__ == "__main__":
    raw_root = "..."
    save_root = "..."

    preprocess_and_save(
        raw_root=raw_root,
        save_root=save_root,
        frames_number=10,
        overwrite=False,
    )