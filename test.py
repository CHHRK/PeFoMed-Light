import os
import torch
import pandas as pd
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from torchvision import transforms
from pefomed.models.pefomed_model import PeFoMedModel
from pefomed.datasets.medical_dataset import MedicalReportDataset


def compute_cider(refs, hyps):
    vec = CountVectorizer().fit(refs + hyps)
    r = vec.transform(refs)
    h = vec.transform(hyps)
    return cosine_similarity(r, h).diagonal().mean()


def collate_fn(batch):
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    imgs = [tf(b["image"]) for b in batch]
    return torch.stack(imgs), [b["text"] for b in batch]


def main():

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")

    test_set = MedicalReportDataset("images", "my_dataset/test/reports.csv")
    loader = torch.utils.data.DataLoader(test_set, batch_size=1,
                                         shuffle=False, collate_fn=collate_fn)

    model = PeFoMedModel(pretrained_path="outputs/best_model_epoch_1.pth").to(device)
    model.eval()

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    preds, refs = [], []

    with torch.no_grad():
        for imgs, texts in tqdm(loader, desc="Testing"):
            imgs = imgs.to(device)

            for i in range(imgs.size(0)):
                preds.append(model.generate(imgs[i]))
                refs.append(texts[i])

    rouge = sum(scorer.score(r, p)['rougeL'].fmeasure for p, r in zip(refs, preds)) / len(refs)
    meteor = sum(meteor_score([word_tokenize(r)], word_tokenize(p)) for p, r in zip(refs, preds)) / len(refs)
    cider = compute_cider(refs, preds)
    em = sum(1 for p, r in zip(preds, refs) if p.lower().strip() == r.lower().strip()) / len(refs)

    print("\nTest Metrics:")
    print("ROUGE-L:", rouge)
    print("METEOR:", meteor)
    print("CIDEr:", cider)
    print("Exact Match:", em)

    pd.DataFrame({"Reference": refs, "Prediction": preds}).to_csv("outputs/test_predictions.csv", index=False)


if __name__ == "__main__":
    main()
