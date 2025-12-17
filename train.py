import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import yaml
from tqdm import tqdm

from pefomed.models.pefomed_model import PeFoMedModel
from pefomed.datasets.medical_dataset import MedicalReportDataset

from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from pycocoevalcap.cider.cider import Cider


def collate_fn(batch):
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    images = [tf(b["image"]) for b in batch]
    texts = [b["text"] for b in batch]
    return {"image": torch.stack(images), "text": texts}


def compute_rouge(preds, refs):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    return sum(scorer.score(r, p)['rougeL'].fmeasure for p, r in zip(preds, refs)) / len(preds)


def compute_meteor(preds, refs):
    return sum(meteor_score([r.split()], p.split()) for p, r in zip(preds, refs)) / len(preds)


def compute_cider(preds, refs):
    g, r = {}, {}
    for i, (p, ref) in enumerate(zip(preds, refs)):
        g[i] = [ref]
        r[i] = [p]
    return Cider().compute_score(g, r)[0]


def compute_em(preds, refs):
    return sum(1 for p, r in zip(preds, refs) if p.strip().lower() == r.strip().lower()) / len(preds)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    preds, refs = [], []

    for batch in tqdm(loader, desc="Validating"):
        imgs = batch["image"].to(device)
        texts = batch["text"]

        for i in range(imgs.size(0)):
            preds.append(model.generate(imgs[i]))
            refs.append(texts[i])

    rouge = compute_rouge(preds, refs)
    meteor = compute_meteor(preds, refs)
    cider = compute_cider(preds, refs)
    em = compute_em(preds, refs)

    print(f"\nValidation Metrics: ROUGE-L {rouge:.4f} METEOR {meteor:.4f} CIDEr {cider:.4f} EM {em:.4f}")
    return rouge, meteor, cider, em


def main():
    cfg = yaml.safe_load(open("configs/pefomed.yaml", "r"))

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )

    train_set = MedicalReportDataset(cfg["data"]["image_root"], cfg["data"]["train_csv"])
    val_set   = MedicalReportDataset(cfg["data"]["image_root"], cfg["data"]["val_csv"])

    train_loader = DataLoader(train_set, batch_size=cfg["train"]["batch_size"],
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model = PeFoMedModel(pretrained_path=cfg["model"]["pretrained_path"]).to(device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(cfg["train"]["learning_rate"])
    )

    os.makedirs(cfg["train"]["output_dir"], exist_ok=True)

    best_rouge = 0

    for epoch in range(cfg["train"]["epochs"]):
        print(f"\nEpoch {epoch+1}/{cfg['train']['epochs']}")
        total_loss = 0

        model.train()
        for batch in tqdm(train_loader, desc="Training"):
            imgs = batch["image"].to(device)
            texts = batch["text"]

            loss = model.compute_loss(imgs, texts)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Avg Loss: {total_loss/len(train_loader):.4f}")

        rouge, meteor, cider, em = validate(model, val_loader, device)

        if rouge > best_rouge:
            best_rouge = rouge
            save_path = f"{cfg['train']['output_dir']}/best_model_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"Saved Best Model → {save_path}")


if __name__ == "__main__":
    main()
