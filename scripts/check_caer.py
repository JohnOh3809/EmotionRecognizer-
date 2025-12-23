import os
from caer_dataset import collect_videos

candidates = []
env_root = os.environ.get("CAER_ROOT")
if env_root:
    candidates.append(env_root)
candidates.extend(["./CAER", "./data/CAER", "./caer/CAER", "./dataset/CAER"]) 

found_any = False
for c in candidates:
    c_abs = os.path.abspath(c)
    if os.path.isdir(c_abs):
        try:
            items = collect_videos(c_abs)
        except Exception as e:
            print(f"Path: {c_abs} -> error while collecting: {e}")
            continue
        print(f"Path: {c_abs} -> videos found: {len(items)}")
        if items:
            print("First 5 items:")
            for it in items[:5]:
                print("  ", it)
        found_any = True
    else:
        print(f"Path: {c_abs} -> not found")

if not found_any:
    print("No CAER dataset found in common locations. Please provide the --data_root path.")
