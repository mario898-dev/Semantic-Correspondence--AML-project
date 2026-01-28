import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class SAMExtractor(nn.Module):
    def __init__(self, repo_dir: str, model_type: str = "vit_b", weights: str = None, device: str = "cuda"):
        """
        Wrapper per l'encoder di Segment Anything (SAM) adatto per Semantic Correspondence.
        Gestisce il caricamento dei pesi, l'interpolazione posizionale dinamica e il fine-tuning parziale.
        """
        super().__init__()
        self.device = device

        # 1. Setup path per importare la libreria esterna segment-anything
        abs_repo_dir = os.path.abspath(repo_dir)
        if abs_repo_dir not in sys.path:
            sys.path.insert(0, abs_repo_dir)

        try:
            from segment_anything import sam_model_registry
        except ImportError as e:
            raise ImportError(f"Errore import SAM da {abs_repo_dir}. Assicurati che la cartella esista. Errore: {e}")

        # 2. Inizializzazione dell'architettura SAM
        # Creiamo il modello senza caricare automaticamente i pesi, 
        # in modo da gestire manualmente il caricamento nella sezione successiva
        print(f"Costruzione architettura SAM ({model_type})...")
        self.model = sam_model_registry[model_type](checkpoint=None)
        self.model.to(device)

        # 3. Caricamento dei pesi del modello pre-addestrato
        if weights and os.path.exists(weights):
            print(f"Caricamento pesi da: {os.path.basename(weights)}")
            try:
                # weights_only=False permette di caricare checkpoint salvati in formati legacy
                # Nota: usare con cautela file da fonti non attendibili
                checkpoint = torch.load(weights, map_location=device, weights_only=False)
            except TypeError:
                # Fallback per versioni di PyTorch che non supportano weights_only
                checkpoint = torch.load(weights, map_location=device)

            # Estrazione dello state_dict: il checkpoint puo' essere un dizionario 
            # con metadati (es. epoca, optimizer) oppure direttamente lo state_dict
            if isinstance(checkpoint, dict) and "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint

            # Pulizia delle chiavi: rimuoviamo il prefisso 'model.' che viene aggiunto
            # quando si salva un modello wrappato in DataParallel o DistributedDataParallel
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("model."):
                    new_state_dict[k.replace("model.", "", 1)] = v
                else:
                    new_state_dict[k] = v
            
            # Caricamento con strict=False per ignorare eventuali chiavi mancanti 
            # dovute a differenze tra versioni del modello
            msg = self.model.load_state_dict(new_state_dict, strict=False)
            print(f"Pesi SAM caricati. Report: {msg}")
        elif weights:
            raise FileNotFoundError(f"File pesi non trovato: {weights}")
        else:
            print("ATTENZIONE: Nessun peso specificato. SAM inizializzato con pesi casuali (utile solo per debug).")

        # 4. Buffer per normalizzazione (ImageNet mean/std per il preprocessing in ingresso)
        self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("imagenet_std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor, extract_layer: int = None) -> torch.Tensor:
        """
        Forward pass dell'image encoder.
        x: (B, 3, H, W) -> Le immagini possono avere size arbitraria.
        Returns: (B, C, H/16, W/16) -> Feature map densa.
        """
        enc = self.model.image_encoder

        # --- A. Preprocessing ---
        # SAM si aspetta input normalizzati internamente basandosi su 0-255.
        # Qui gestiamo sia input normalizzati ImageNet (tipici di Dataloader standard) sia [0,1].
        if x.max() > 2.0 or x.min() < -1.0:
            # Assumiamo standardizzazione ImageNet -> Denormalizziamo a [0,1]
            mean = self.imagenet_mean.to(x.device)
            std = self.imagenet_std.to(x.device)
            x01 = (x * std + mean).clamp(0.0, 1.0)
        else:
            # Assumiamo già in [0,1]
            x01 = x.clamp(0.0, 1.0)

        # Scaling a 0-255 e normalizzazione specifica di SAM
        x255 = x01 * 255.0
        x_sam = (x255 - self.model.pixel_mean) / self.model.pixel_std

        # --- B. Gestione Positional Embeddings Dinamica ---
        # SAM è trainato su 1024x1024. Se l'immagine è diversa, interpoliamo i pos_embed.
        # CRUCIALE: Non sovrascriviamo enc.pos_embed, usiamo una variabile temporanea.
        patch_size = enc.patch_embed.proj.stride[0] # Di solito 16
        hp, wp = x_sam.shape[-2] // patch_size, x_sam.shape[-1] // patch_size

        # Reference ai pesi originali
        pos_embed = enc.pos_embed
        
        # Interpolazione se le dimensioni dei patch non coincidono con quelle native (64x64)
        if pos_embed.shape[1] != hp or pos_embed.shape[2] != wp:
            pos_embed = pos_embed.permute(0, 3, 1, 2)  # (1, C, H, W)
            pos_embed = F.interpolate(pos_embed, size=(hp, wp), mode="bilinear", align_corners=False)
            pos_embed = pos_embed.permute(0, 2, 3, 1)  # (1, H, W, C)

        # --- C. Estrazione Feature ---
        # 1. Patch Embedding
        out = enc.patch_embed(x_sam)
        
        # 2. Somma Positional Embedding (usando il tensore interpolato)
        if enc.pos_embed is not None:
            out = out + pos_embed 

        # 3. Passaggio attraverso i blocchi Transformer
        limit = len(enc.blocks) if extract_layer is None else extract_layer + 1

        for i, blk in enumerate(enc.blocks):
            out = blk(out)
            if i == limit - 1:
                break
        
        # Output shape: (B, H_patch, W_patch, C) -> Permute to (B, C, H_patch, W_patch)
        feats = out.permute(0, 3, 1, 2)
        return feats

    def setup_finetuning(self, num_layers: int):
        """
        Configura i layer per il fine-tuning.
        Congela tutto l'encoder, poi sblocca gli ultimi 'num_layers' blocchi.
        """
        # 1. Congela tutto inizialmente
        for param in self.model.image_encoder.parameters():
            param.requires_grad = False

        blocks = self.model.image_encoder.blocks
        total_blocks = len(blocks)
        
        # 2. Sblocca gli ultimi k layer
        if num_layers > 0:
            start_layer = total_blocks - num_layers
            if start_layer < 0: start_layer = 0
            
            for i in range(start_layer, total_blocks):
                for param in blocks[i].parameters():
                    param.requires_grad = True
            
            print(f"{self.__class__.__name__}: Sbloccati gli ultimi {num_layers}/{total_blocks} blocchi per il fine-tuning.")
        else:
            print(f"{self.__class__.__name__}: Encoder completamente congelato (nessun parametro addestrabile).")
