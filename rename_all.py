import os

def rename_subfolders(parent_dir):
    base = os.path.basename(os.path.abspath(parent_dir))

    for name in os.listdir(parent_dir):
        old_path = os.path.join(parent_dir, name)
        if os.path.isdir(old_path):
            new_name = f"{base}-{name}"
            new_path = os.path.join(parent_dir, new_name)

            # avoid collisions
            if not os.path.exists(new_path):
                os.rename(old_path, new_path)
                print(f"Renamed: {old_path} -> {new_path}")
            else:
                print(f"Skipped (already exists): {new_path}")

rename_subfolders(r"C:\Users\AI2\Downloads\test\nir")
rename_subfolders(r"C:\Users\AI2\Downloads\test\vis")
rename_subfolders(r"C:\Users\AI2\Downloads\test\rednir")