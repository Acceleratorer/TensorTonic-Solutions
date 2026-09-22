import math
import torch
def gated_mla(
      hidden_states: torch.Tensor,
      query_projection: torch.Tensor,
      latent_down_projection: torch.Tensor,
      key_up_projection: torch.Tensor,
      value_up_projection: torch.Tensor,
      output_gate_projection: torch.Tensor,
      output_projection: torch.Tensor,
      num_heads: int,
      causal: bool = True,
  ) -> dict[str, torch.Tensor]:
      batch, seq_len, model_dim = hidden_states.shape
      head_dim = model_dim // num_heads

      latent_cache = torch.matmul(
          hidden_states,
          latent_down_projection.transpose(-1, -2),
      )

      queries = torch.matmul(
          hidden_states,
          query_projection.transpose(-1, -2),
      )

      keys = torch.matmul(
          latent_cache,
          key_up_projection.transpose(-1, -2),
      )

      values = torch.matmul(
          latent_cache,
          value_up_projection.transpose(-1, -2),
      )

      queries = queries.view(batch, seq_len, num_heads, head_dim)
      keys = keys.view(batch, seq_len, num_heads, head_dim)
      values = values.view(batch, seq_len, num_heads, head_dim)

      queries = queries.transpose(1, 2)
      keys = keys.transpose(1, 2)
      values = values.transpose(1, 2)

      scores = torch.matmul(
          queries,
          keys.transpose(-1, -2),
      ) / math.sqrt(head_dim)

      if causal:
          mask = torch.triu(
              torch.ones(
                  seq_len,
                  seq_len,
                  dtype=torch.bool,
                  device=hidden_states.device,
              ),
              diagonal=1,
          )
          scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)

      attention = torch.softmax(scores, dim=-1)
      context = torch.matmul(attention, values)

      context = context.transpose(1, 2).contiguous()
      context = context.view(batch, seq_len, model_dim)

      gate = torch.sigmoid(
          torch.matmul(
              hidden_states,
              output_gate_projection.transpose(-1, -2),
          )
      )

      output = torch.matmul(
          gate * context,
          output_projection.transpose(-1, -2),
      )

      return {
          "output": output,
          "latent_cache": latent_cache,
      }