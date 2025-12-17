import os
import torch
import torch.nn as nn
from transformers import CLIPModel, GPT2LMHeadModel, AutoTokenizer
from peft import LoraConfig, get_peft_model


class PeFoMedModel(nn.Module):
    def __init__(self, pretrained_path=None):
        super().__init__()

        clip = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14",
            torch_dtype=torch.float32,
            local_files_only=False
        )

        self.vision_encoder = clip.vision_model.to("cpu")

        for p in self.vision_encoder.parameters():
            p.requires_grad = False
        self.vision_encoder.eval()

        
        # Grouped tokens = 4 × 1024 = 4096
        self.up_proj = nn.Linear(4096, 5632)     # 4096 → 5632
        self.img_to_gpt2 = nn.Linear(5632, 768)  # 5632 → GPT-2 hidden

        
        self.text_decoder = GPT2LMHeadModel.from_pretrained("gpt2")

        # Freeze GPT-2 base
        for p in self.text_decoder.parameters():
            p.requires_grad = False

        
        lora_cfg = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["c_attn", "c_proj"],
            task_type="CAUSAL_LM",
        )
        self.text_decoder = get_peft_model(self.text_decoder, lora_cfg)

        
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        
        if pretrained_path and os.path.exists(pretrained_path):
            state = torch.load(pretrained_path, map_location="cpu")
            self.load_state_dict(state, strict=False)
            print(f"Loaded pretrained weights from {pretrained_path}")

    
    def encode_image(self, images):
        """
        Extract last_hidden_state (1024 dim), take 4 tokens,
        concatenate to 4096, then run projections.
        """
        with torch.no_grad():
            out = self.vision_encoder(images, output_hidden_states=False)
            tokens = out.last_hidden_state           # (B, SeqLen, 1024)

        
        group = tokens[:, 1:5, :]                   

        
        grouped = group.reshape(images.size(0), 4096)

       
        x = torch.relu(self.up_proj(grouped))      
        x = self.img_to_gpt2(x)                     

        return x

 
    def forward(self, images, input_ids, attention_mask):
        B = images.size(0)

        
        img_embed = self.encode_image(images).unsqueeze(1)

        
        text_embeds = self.text_decoder.transformer.wte(input_ids)

        
        inputs_embeds = torch.cat([img_embed, text_embeds], dim=1)  # (B, 1+T, 768)

       
        new_mask = torch.cat(
            [torch.ones(B, 1).to(attention_mask.device), attention_mask],
            dim=1
        )

        
        labels = input_ids.clone()
        labels = torch.cat(
            [torch.full((B, 1), -100).to(labels.device), labels],
            dim=1
        )

        
        outputs = self.text_decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=new_mask,
            labels=labels
        )

        return outputs

    
    def compute_loss(self, images, reports):
        device = next(self.parameters()).device

        encoded = self.tokenizer(
            reports,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(device)

        outputs = self.forward(
            images,
            encoded["input_ids"],
            encoded["attention_mask"]
        )
        return outputs.loss

    
    @torch.no_grad()
    def generate(self, image, max_new_tokens=80):
        self.eval()
        device = next(self.parameters()).device

        if image.dim() == 3:
            image = image.unsqueeze(0)
        image = image.to(device)

        # Image prefix token
        img_emb = self.encode_image(image).unsqueeze(1)

        # Prompt
        prompt = "Findings:"
        ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        txt_emb = self.text_decoder.transformer.wte(ids)

        full_embeds = torch.cat([img_emb, txt_emb], dim=1)

        out = self.text_decoder.generate(
            inputs_embeds=full_embeds,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )

        return self.tokenizer.decode(out[0], skip_special_tokens=True)
