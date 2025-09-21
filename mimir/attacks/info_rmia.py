from dataclasses import dataclass
import pdb
from typing import Optional, List, cast
import numpy as np
import torch
from torch.nn.functional import kl_div

from .all_attacks import Attack
from mimir.models import ReferenceModel


@dataclass
class InfoRMIAParams:
    """Lightweight config for Info-RMIA aggregation behavior."""

    aggregate: str = "mean"  # "mean" or "sum"


class InfoRMIAToken(Attack):
    """
    Token-based Info-RMIA that conforms to the Attack API used by run.py.

    Score(x) = aggregate_t [ log p_target(x_t | x_<t) - log p_ref(x_t | x_<t) ]

    Implementation notes:
    - We reuse the target tokenizer to create token ids, and feed those ids to the
      reference model if and only if the vocabularies match (assumed true for same-family models like Pythia).
    - We aggregate across valid next-token positions (length - 1), matching get_probabilities behavior.
    """

    def __init__(
        self,
        config,
        target_model,
        ref_model: Optional[ReferenceModel] = None,
        is_blackbox: bool = True,
    ):
        super().__init__(config, target_model, ref_model, is_blackbox)
        self.params = InfoRMIAParams(
            aggregate=getattr(config, "info_rmia_aggregate", "mean")
        )

        # Store multiple reference models
        self.ref_models = []
        
        # If a single ref_model is provided, use it
        if ref_model is not None:
            self.ref_models.append(ref_model)
        # Otherwise, create reference models from config
        elif (config.ref_config and config.ref_config.models):
            for model_name in config.ref_config.models:
                self.ref_models.append(ReferenceModel(config, model_name))
        
        if not self.ref_models:
            raise ValueError("InfoRMIAToken requires at least one reference model")

    def _aggregate(self, diffs: List[float]) -> float:
        if len(diffs) == 0:
            return float("nan")
        if self.params.aggregate == "sum":
            return float(sum(diffs))
        elif self.params.aggregate == "mink":
            k = self.params.k if hasattr(self.params, 'k') else 0.2
            return float(np.mean(sorted(diffs)[: int(len(diffs) * k)]))
        elif self.params.aggregate == "mean":
            return float(sum(diffs) / len(diffs))
        else:
            raise ValueError(f"Unknown aggregation method: {self.params.aggregate}")

    def _attack(self, document: str, probs: List[float], tokens=None, **kwargs):
        # Get target model outputs (per-token log-probs and optionally full distributions)
        target_log_px, target_log_all = self.target_model.get_probabilities(
            document, return_all_probs=True
        )

        # Normalize target token log-probs to tensor
        if isinstance(target_log_px, torch.Tensor):
            target_log_px_t = target_log_px.clone()
        else:
            target_log_px_t = torch.tensor(np.asarray(target_log_px))

        # Normalize target full distributions if present
        target_log_all_t = None
        if target_log_all is not None:
            if isinstance(target_log_all, torch.Tensor):
                target_log_all_t = target_log_all.clone()
            else:
                target_log_all_t = torch.tensor(target_log_all)
            # remove leading batch dim if present
            if target_log_all_t.ndim == 3 and target_log_all_t.shape[0] == 1:
                target_log_all_t = target_log_all_t.squeeze(0)

        # Determine device to use (prefer target full-dist device when available)
        if isinstance(target_log_all_t, torch.Tensor):
            target_device = target_log_all_t.device
        elif isinstance(target_log_px_t, torch.Tensor) and target_log_px_t.device is not None:
            target_device = target_log_px_t.device
        else:
            target_device = torch.device("cpu")

        target_log_px_t = target_log_px_t.to(target_device)
        if isinstance(target_log_all_t, torch.Tensor):
            target_log_all_t = target_log_all_t.to(target_device)

        # Collect ref outputs
        ref_px_tensors = []
        ref_all_tensors = []
        for ref_model in self.ref_models:
            try:
                ref_px, ref_all = ref_model.get_probabilities(document, return_all_probs=True)

                # per-token log-probs -> tensor on target device
                ref_px_t = torch.tensor(np.asarray(ref_px), device=target_device)

                # full distributions if present -> tensor (squeeze batch dim)
                ref_all_t = None
                if ref_all is not None:
                    if isinstance(ref_all, torch.Tensor):
                        ref_all_t = ref_all.clone()
                    else:
                        ref_all_t = torch.tensor(ref_all)
                    if ref_all_t.ndim == 3 and ref_all_t.shape[0] == 1:
                        ref_all_t = ref_all_t.squeeze(0)
                    ref_all_t = ref_all_t.to(target_device)

                ref_px_tensors.append(ref_px_t)
                ref_all_tensors.append(ref_all_t)
            except Exception as e:
                print(f"WARNING: reference model {ref_model.name} failed: {e}")
                continue

        if len(ref_px_tensors) == 0:
            print("ERROR: No compatible reference models found!")
            return float("nan")

        # Truncate per-token vectors to minimum length across target and refs
        min_tokens = min([target_log_px_t.shape[0]] + [r.shape[0] for r in ref_px_tensors])
        target_px_trunc = target_log_px_t[:min_tokens]
        ref_px_stack = torch.stack([r[:min_tokens] for r in ref_px_tensors], dim=0)
        avg_ref_log_px = torch.mean(ref_px_stack, dim=0)

        # Compute KL term only if full distributions exist for target and ALL refs and shapes match
        kl_term = None
        if (
            isinstance(target_log_all_t, torch.Tensor)
            and all(isinstance(r, torch.Tensor) for r in ref_all_tensors)
            and all(r is not None for r in ref_all_tensors)
        ):
            shapes_match = all(r.shape == target_log_all_t.shape for r in ref_all_tensors)
            if shapes_match:
                # truncate sequence length to min across distributions (should be equal)
                seq_len = target_log_all_t.shape[0]
                # average reference full distributions
                ref_all_stack = torch.stack([r for r in ref_all_tensors], dim=0)
                avg_ref_all = torch.mean(ref_all_stack, dim=0)
                # ensure on device
                avg_ref_all = avg_ref_all.to(target_device)
                target_all_trunc = target_log_all_t[:min_tokens]
                avg_ref_all_trunc = avg_ref_all[:min_tokens]
                kl_term = torch.sum(avg_ref_all_trunc.exp() * (target_all_trunc - avg_ref_all_trunc), dim=-1)
            else:
                raise ValueError(f"Reference full distributions have mismatched shapes compared to target: {[r.shape for r in ref_all_tensors]} vs {target_log_all_t.shape}")

        if kl_term is None:
            # kl_term = torch.zeros_like(avg_ref_log_px, device=target_device)
            raise ValueError(f"KL term could not be computed due to missing or incompatible full distributions. Failure reason: {isinstance(target_log_all_t, torch.Tensor)}, {all(isinstance(r, torch.Tensor) for r in ref_all_tensors)}, {all(r is not None for r in ref_all_tensors)}, {shapes_match if 'shapes_match' in locals() else 'N/A'}")

        # per-token score and aggregation
        per_token_scores = (target_px_trunc - avg_ref_log_px) - kl_term[: min_tokens]

        # convert to python list of floats for _aggregate
        scores_list = per_token_scores.detach().cpu().numpy().tolist()
        return -self._aggregate(scores_list)


class InfoRMIASeq(InfoRMIAToken):
    def __init__(
        self,
        config,
        target_model,
        ref_model: Optional[ReferenceModel] = None,
        is_blackbox: bool = True,
    ):
        super().__init__(config, target_model, ref_model, is_blackbox)

    def _attack(self, document: str, probs: List[float], tokens=None, **kwargs):
        target_logp = np.mean(probs)
        
        # Get log probs from all reference models and average them
        ref_logps = []
        for ref_model in self.ref_models:
            ref_logps.append(np.mean(ref_model.get_probabilities(document)))
        
        ref_logp = np.mean(ref_logps) if ref_logps else float("nan")
        
        return -(target_logp - ref_logp)
