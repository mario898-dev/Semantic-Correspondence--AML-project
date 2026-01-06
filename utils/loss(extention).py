import torch
import torch.nn as nn
import torch.nn.functional as F

class WindowSoftTargetLoss(nn.Module):
    def __init__(self, sigma=2.0, temperature=0.1):
        """
        Loss basata su KL Divergence con target Gaussiano (Window Soft Target).
        
        Args:
            sigma: Deviazione standard per la Gaussiana del target (in coordinate feature grid).
            temperature: Temperatura per scalare i logit prima della Softmax.
        """
        super().__init__()
        self.sigma = sigma
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction="batchmean")

    def forward(self, sim_logits, trg_kps, img_size, feature_shape):
        """
        Args:
            sim_logits: (N_kps, Hf*Wf) Logits di similarità predetti (output di compute_similarity_logits).
            trg_kps: (N_kps, 2) Keypoints target validi (Ground Truth) in pixel.
            img_size: Dimensione originale dell'immagine (int, es. 480 o 518).
            feature_shape: Tuple (Hf, Wf) dimensioni spaziali delle feature.
        
        Returns:
            loss: Scalare (KL Divergence).
        """
        device = sim_logits.device
        N_kps = sim_logits.shape[0]
        Hf, Wf = feature_shape
        
        if N_kps == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # 1. Calcolo stride e conversione coordinate pixel -> feature grid
        # Stride calcolato come rapporto tra img_size e feature dim
        stride_x = img_size / Wf
        stride_y = img_size / Hf
        
        # Patch center convention: pixel = (idx + 0.5) * stride
        # => idx = pixel / stride - 0.5
        trg_x = trg_kps[:, 0] / stride_x - 0.5
        trg_y = trg_kps[:, 1] / stride_y - 0.5
        
        # 2. Creazione della griglia coordinate (Hf, Wf)
        # x_grid: (Wf,), y_grid: (Hf,)
        x_grid = torch.arange(Wf, device=device).float()
        y_grid = torch.arange(Hf, device=device).float()
        
        # Meshgrid: yy (Hf, Wf), xx (Hf, Wf)
        yy, xx = torch.meshgrid(y_grid, x_grid, indexing='ij')
        
        # Flatten per vettorizzazione: (1, Hf*Wf)
        xx = xx.reshape(1, -1)
        yy = yy.reshape(1, -1)
        
        # Reshape target per broadcasting: (N_kps, 1)
        trg_x = trg_x.unsqueeze(1)
        trg_y = trg_y.unsqueeze(1)
        
        # 3. Generazione Gaussiana Target (Soft Target)
        # Distanza euclidea quadrata: (x - mux)^2 + (y - muy)^2
        dist_sq = (xx - trg_x)**2 + (yy - trg_y)**2
        
        # Heatmap Gaussiana: exp(-dist^2 / (2*sigma^2))
        target_heatmap = torch.exp(-dist_sq / (2 * self.sigma**2))
        
        # Normalizzazione: la somma deve essere 1 (distribuzione di probabilità)
        target_heatmap = target_heatmap / (target_heatmap.sum(dim=1, keepdim=True) + 1e-6)
        
        # 4. Predizioni (Log-Probabilities)
        # Applichiamo LogSoftmax ai logit scalati dalla temperatura
        log_probs = F.log_softmax(sim_logits / self.temperature, dim=1)
        
        # 5. Calcolo Loss
        loss = self.kl_div(log_probs, target_heatmap)
        
        return loss
