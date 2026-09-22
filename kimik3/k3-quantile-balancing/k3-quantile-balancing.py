import torch

def quantile_balancing(
      router_scores: torch.Tensor,
      current_bias: torch.Tensor,
      selected_count: int,
  ) -> dict[str, torch.Tensor]:
      tokens, experts = router_scores.shape

      biased_scores = router_scores + current_bias
      sorted_indices = torch.argsort(
          biased_scores,
          dim=1,
          descending=True,
          stable=True,
      )

      selected_experts = sorted_indices[:, :selected_count]

      selected_scores = torch.gather(
          router_scores,
          dim=1,
          index=selected_experts,
      )
      mixture_weights = selected_scores / selected_scores.sum(
          dim=1,
          keepdim=True,
      )

      expert_loads = torch.bincount(
          selected_experts.reshape(-1),
          minlength=experts,
      )

      cutoff = torch.gather(
          biased_scores,
          dim=1,
          index=sorted_indices[:, selected_count:selected_count + 1],
      ).squeeze(1)

      target_load = tokens * selected_count // experts

      margins = router_scores - cutoff.unsqueeze(1)
      sorted_margins = torch.sort(
          margins,
          dim=0,
          descending=True,
          stable=True,
      ).values

      quantiles = sorted_margins[target_load]
      next_bias = -quantiles
      next_bias = next_bias - next_bias.mean()

      return {
          "selected_experts": selected_experts,
          "mixture_weights": mixture_weights,
          "expert_loads": expert_loads,
          "next_bias": next_bias,
      }