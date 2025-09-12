import torch
import os

data_dir = r"C:\Users\Akshit\OneDrive\Documents\TERM 5\IPD\MGTAB\MGTAB"

for name in ["features.pt", "edge_index.pt", "labels_bot.pt", "labels_stance.pt"]:
    path = os.path.join(data_dir, name)
    obj = torch.load(path, map_location="cpu")
    print(name, type(obj))
