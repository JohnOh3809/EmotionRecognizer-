from caer_dataset import collect_videos
from collections import Counter
items = collect_videos(r"C:\Users\purpl\Downloads\caer\CAER")
print('total', len(items))
print('counts', Counter([s for _, s, _, _ in items]))
print('sample', items[:12])
