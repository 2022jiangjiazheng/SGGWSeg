import os
import shutil
import random
from tqdm import tqdm


def split_deepglobe_dataset(src_dir, dest_dir, train_count=4696, test_count=1530):
    """Randomly split paired DeepGlobe images and masks into train/test sets.

    Args:
        src_dir: Source directory containing the original paired files.
        dest_dir: Destination dataset directory.
        train_count: Number of pairs assigned to the training set.
        test_count: Number of pairs assigned to the test set.
    """
    # 1. Create the destination directory structure.
    sub_dirs = [
        'images/train', 'images/test',
        'zones/train', 'zones/test'
    ]
    for sd in sub_dirs:
        os.makedirs(os.path.join(dest_dir, sd), exist_ok=True)

    # 2. Extract base IDs (for example, 104_sat.jpg -> 104).
    # Satellite images define the candidate IDs; masks are checked before copy.
    all_files = os.listdir(src_dir)
    base_ids = [f.split('_sat.jpg')[0] for f in all_files if f.endswith('_sat.jpg')]

    print(f"Detected {len(base_ids)} source samples.")

    if len(base_ids) < (train_count + test_count):
        print(
            f"Warning: {len(base_ids)} source samples are insufficient for "
            f"the requested {train_count + test_count} samples."
        )
        return

    # 3. Shuffle reproducibly and create the split.
    random.seed(42)
    random.shuffle(base_ids)

    train_ids = base_ids[:train_count]
    test_ids = base_ids[train_count: train_count + test_count]

    # 4. Copy paired files into their split directories.
    def process_split(ids, split_name):
        print(f"Processing the {split_name} split...")
        for b_id in tqdm(ids):
            # Source paths.
            sat_src = os.path.join(src_dir, f"{b_id}_sat.jpg")
            mask_src = os.path.join(src_dir, f"{b_id}_mask.png")

            # Images and masks are stored under images and zones, respectively.
            sat_dst = os.path.join(dest_dir, 'images', split_name, f"{b_id}.jpg")
            mask_dst = os.path.join(dest_dir, 'zones', split_name, f"{b_id}.png")

            # Copy only complete image/mask pairs.
            if os.path.exists(sat_src) and os.path.exists(mask_src):
                shutil.copy(sat_src, sat_dst)
                shutil.copy(mask_src, mask_dst)
            else:
                print(f"Missing paired file(s): {b_id}")

    process_split(train_ids, 'train')
    process_split(test_ids, 'test')

    print("\nDataset split complete.")
    print(f"Output: {os.path.abspath(dest_dir)}")


if __name__ == "__main__":
    # Update these paths when the source and output are stored elsewhere.
    SOURCE = "raw_data"
    TARGET = "data"

    split_deepglobe_dataset(SOURCE, TARGET)
