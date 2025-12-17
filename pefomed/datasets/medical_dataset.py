import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image


class MedicalReportDataset(Dataset):
    def __init__(self, image_root, csv_path):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(csv_path)

        df = pd.read_csv(csv_path)
        self.samples = []

        for _, row in df.iterrows():
            if "Path" not in row or "Report_Impression" not in row:
                continue

            rel = row["Path"].strip()

            # remove leading "images/"
            if rel.startswith("images/"):
                rel = rel[7:]

            img_path = os.path.join(image_root, rel)

            if os.path.exists(img_path):
                self.samples.append((img_path, row["Report_Impression"]))

        print(f"Loaded {len(self.samples)} samples from {csv_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, text = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return {"image": img, "text": text}
