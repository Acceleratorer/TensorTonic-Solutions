import torch


def _read_depth_sources(sources, pseudo_query, eps):
      normalized = sources / torch.sqrt(
          sources.square().mean(dim=-1, keepdim=True) + eps
      )
      logits = (normalized * pseudo_query).sum(dim=-1)
      weights = torch.softmax(logits, dim=0)
      retrieved = (weights.unsqueeze(-1) * sources).sum(dim=0)
      return retrieved, weights


def block_attention_residual(
      embedding: torch.Tensor,
      previous_outputs: torch.Tensor,
      pseudo_query: torch.Tensor,
      block_size: int,
      eps: float = 1e-6,
  ) -> dict[str, torch.Tensor]:
      sources = [embedding]
      num_layers = previous_outputs.shape[0]

      complete_layers = (num_layers // block_size) * block_size

      for start in range(0, complete_layers, block_size):
          block = previous_outputs[start:start + block_size].sum(dim=0)
          sources.append(block)

      if complete_layers < num_layers:
          partial = previous_outputs[complete_layers:].sum(dim=0)
          sources.append(partial)

      block_sources = torch.stack(sources, dim=0)

      retrieved, weights = _read_depth_sources(
          block_sources,
          pseudo_query,
          eps,
      )

      return {
          "retrieved_representation": retrieved,
          "attention_weights": weights,
          "block_sources": block_sources,
      }