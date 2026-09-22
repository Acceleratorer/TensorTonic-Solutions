import torch
def kda_recurrence(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, decay_logits:
  torch.Tensor, write_strength: torch.Tensor, output_gate_logits: torch.Tensor,
  output_projection: torch.Tensor, initial_state: torch.Tensor, g_min: float = -5.0, eps: float
  = 1e-6) -> dict[str, torch.Tensor]:
      state = initial_state.clone()
      batch, seq_len, heads, key_dim = query.shape
      value_dim = value.shape[-1]
      identity = torch.eye(key_dim, dtype=state.dtype, device=state.device).view(1, 1, key_dim,
      key_dim)
      outputs = []

      for t in range(seq_len):
          q = query[:, t]
          k = key[:, t]
          v = value[:, t]

          alpha = torch.exp(g_min * torch.sigmoid(decay_logits[:, t]))
          beta = write_strength[:, t].reshape(batch, heads, 1, 1)

          decayed = alpha.unsqueeze(-1) * state
          key_outer = k.unsqueeze(-1) * k.unsqueeze(-2)

          state = torch.matmul(identity - beta * key_outer, decayed)
          state = state + beta * k.unsqueeze(-1) * v.unsqueeze(-2)

          read = torch.einsum("bhdv,bhd->bhv", state, q)
          rms = torch.sqrt(read.square().mean(dim=-1, keepdim=True) + eps)
          read = read / rms
          read = read * torch.sigmoid(output_gate_logits[:, t])
          outputs.append(read)

      outputs = torch.stack(outputs, dim=1)
      outputs = outputs.reshape(batch, seq_len, heads * value_dim)
      outputs = torch.matmul(outputs, output_projection.transpose(-1, -2))

      return {"outputs": outputs, "final_state": state}