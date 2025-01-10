import torch
from abc import ABC, abstractmethod

def get_capacity(num_tokens: int, num_experts: int, capacity_factor: float, min_capacity=None):
    """
    Calculate the capacity of each expert.

    Args:
        num_tokens (int): num of the input tokens.
        num_experts (int): num of the experts.
        capacity_factor (float): Capacity factor.
        min_capacity (int, optional): Minimum capacity. Defaults to None.

    Returns:
        Tensor: Capacity of each expert.
    """
    import math
    capacity = math.ceil((num_tokens / num_experts) * capacity_factor)
    if min_capacity is not None and capacity < min_capacity:
        capacity = min_capacity
    return capacity

class MoEAuxLossAutoScaler(torch.autograd.Function):
    """An AutoScaler that compute and scales the grad for auxiliary loss."""

    main_loss_backward_scale: torch.Tensor = torch.tensor(1.0)

    @staticmethod
    def forward(ctx, output: torch.Tensor, aux_loss: torch.Tensor):
        """Preserve the aux_loss by storing it in the context to avoid garbage collection.

        Args:
            output (torch.Tensor): The output tensor.
            aux_loss (torch.Tensor): The auxiliary loss tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        ctx.save_for_backward(aux_loss)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Compute and scale the gradient for auxiliary loss..

        Args:
            grad_output (torch.Tensor): The gradient of the output.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The gradient of the output, scaled auxiliary loss gradient.
        """
        (aux_loss,) = ctx.saved_tensors
        aux_loss_backward_scale = MoEAuxLossAutoScaler.main_loss_backward_scale
        scaled_aux_loss_grad = torch.ones_like(aux_loss) * aux_loss_backward_scale
        return grad_output, scaled_aux_loss_grad

    @staticmethod
    def set_loss_scale(scale: torch.Tensor):
        """set the scale of the aux loss.

        Args:
            scale (torch.Tensor): The scale value to set. Please ensure that the scale passed in matches the scale of the main_loss.
        """
        MoEAuxLossAutoScaler.main_loss_backward_scale = scale


def topk_softmax_with_capacity(
    logits: torch.Tensor,
    topk: int,
    capacity_factor: float = None,
    min_capacity: float = None,
    pad_to_capacity: bool = False,
    drop_policy: str = "probs",
    use_pre_softmax: bool = False,
):
    """Apply capacity and padding to the top-k selection.
    Args:
        logits (torch.Tensor): Logits tensor.
        topk (int): The number of experts to select for each token.
        capacity_factor (int): The capacity factor of each expert. Will drop tokens if the number of tokens exceeds the capacity.
        pad_to_capacity (bool): Whether to need padding in token drop mode.
        drop_policy (str): The policy to drop tokens. Can be either "prob" or "position". If "prob", the tokens with the lowest probabilities will be dropped. If "position", tokens at the end of each batch will be dropped.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Probs, indices and tokens_per_expert tensor.

        (1) If there's no token padding, the shape of probs and indices is [tokens, top_k], indicating the selected experts for each token.
        (2) If there's token padding, the shape of probs and indices is [num_expert, capacity], indicating the tokens selected for each expert.
    """
    assert logits.dim() == 2, f"Expected 2D logits [num_tokens, num_experts], got {logits.dim()}."
    num_tokens = logits.shape[0]
    num_experts = logits.shape[1]
    if use_pre_softmax:
        # Pre softmax
        scores = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(logits)
        probs, top_indices = torch.topk(scores, k=topk, dim=1)
    else:
        # Post softmax
        if topk == 1:
            # Requires applying softmax before selecting the top-k when k is 1, since softmax on a [num_tokens, 1] would yield a zero gradient.
            raise ValueError("Please use --moe-router-pre-softmax when topk is 1.")
        scores, top_indices = torch.topk(logits, k=topk, dim=1)
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).type_as(logits)

    if capacity_factor is None:
        # TopK without capacity
        tokens_per_expert = torch.bincount(top_indices.view(-1), minlength=num_experts)
        return probs, top_indices, tokens_per_expert
    else:
        # TopK with capacity
        expert_capacity = get_capacity(
            num_tokens=num_tokens * topk,
            num_experts=num_experts,
            capacity_factor=capacity_factor,
        )
        # TopK selection, Maskout unused experts
        topk_masked_gates = torch.zeros_like(logits).scatter(1, top_indices, probs)
        topk_mask = torch.zeros_like(logits).scatter(1, top_indices, 1)

        # Maskout exceeded tokens
        if drop_policy == "probs":
            # 使用 torch.topk 按列（dim=0）选取概率最大的 expert_capacity 个 token。
            # capacity_probs：大小为 [expert_capacity, num_experts]，表示每个专家中选中的 top-k token 的概率。
            # capacity_indices：大小为 [expert_capacity, num_experts]，表示选中 token 的索引。
            capacity_probs, capacity_indices = torch.topk(
                topk_masked_gates, k=expert_capacity, dim=0, sorted=False
            )
            capacity_mask = torch.zeros_like(logits).scatter(0, capacity_indices, 1)
        elif drop_policy == "position":
            # cpu
            _, capacity_indices = torch.topk(topk_mask, k=expert_capacity, dim=0, sorted=True)

            capacity_mask = torch.zeros_like(logits).scatter(0, capacity_indices, 1)
            capacity_probs = torch.gather(topk_masked_gates, 0, capacity_indices)
        else:
            raise ValueError(f"Invalid drop_policy: {drop_policy}")

        if pad_to_capacity:
            final_probs, final_indices = (
                capacity_probs.T.contiguous(),
                capacity_indices.T.contiguous(),
            )
            tokens_per_expert_before_capacity = topk_mask.sum(dim=0)
        else:
            # Get exceed mask and maskout exceeded probs and indices
            # final_mask：[num_tokens, num_experts]，标记每个 token 是否符合容量限制
            # drop_mask：标记超出容量限制的 token
            # exceed_mask：[num_tokens, topk]，针对每个 token 的 top-k 专家，标记哪些分配超出容量限制
            # final_probs，将超出限制的概率值设置为 0
            # final_indices，将超出限制的专家索引设置为无效值。
            final_mask = torch.logical_and(topk_mask, capacity_mask)
            drop_mask = torch.logical_not(final_mask)
            exceed_mask = torch.gather(drop_mask, 1, top_indices)
            final_probs = probs * torch.logical_not(exceed_mask)
            final_indices = top_indices.clone().masked_fill_(
                exceed_mask, torch.iinfo(torch.long).max
            )
            tokens_per_expert_before_capacity = topk_mask.sum(dim=0)
        return final_probs, final_indices, tokens_per_expert_before_capacity, topk_masked_gates, final_mask

# 计算负载均衡损失
def switch_load_balancing_loss_func(
    probs: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    topk: int,
    moe_aux_loss_coeff: float,
    sequence_partition_group=None,
):
    """Calculate the auxiliary loss for load balancing.
    Refer to the Switch Transformer paper (https://arxiv.org/abs/2101.03961) for details.

    Args:
        probs (torch.Tensor): Softmax probabilities output by the router for each token. [num_tokens, num_experts]
        tokens_per_expert (torch.Tensor): Number of tokens assigned to each expert. [num_experts]
        topk (int): The number of experts selected for each token.
        moe_aux_loss_coeff (float): The coefficient for the auxiliary loss.
        sequence_partition_group (optional): The parallel group over which the sequence is partitioned. If None, no partitioning is applied. Defaults to None.

    Returns:
        torch.Tensor: The auxiliary loss for load balancing.
    """
    num_sub_sequence = 1

    # If the sequence is partitioned by certain parallelism strategies like Sequence Parallelism or Context Parallelism, compute the gradient of the auxiliary loss with respect to the full sequence.
    # if sequence_partition_group is not None:
    #     # We can keep `aggregated_probs_per_expert` local since we don't need the gradient for `tokens_per_expert`, saving one allreduce operation for `aggregated_probs_per_expert`.
    #     num_sub_sequence = torch.distributed.get_world_size(sequence_partition_group)
    #     torch.distributed.all_reduce(tokens_per_expert, group=sequence_partition_group)

    num_tokens = probs.shape[0] * num_sub_sequence
    num_experts = probs.shape[1]

    # The formula of aux_loss: aux_loss = sum((probs_per_expert/num_tokens) * (tokens_per_expert/(num_tokens*topk))) * num_experts * moe_aux_loss_coeff.
    # This can be simplified to fuse the division and multiplication operations.
    # aggregated_probs_per_expert 代表了每个专家在整个序列中被选择的频率，代表了每个专家的激活概率
    aggregated_probs_per_expert = probs.sum(dim=0)
    # aggregated_probs_per_expert * tokens_per_expert 计算每个专家的“负担”
    # torch.sum(...)：对所有专家的负担求和，得到整体的负担值。
    aux_loss = torch.sum(aggregated_probs_per_expert * tokens_per_expert) * (
        num_experts * moe_aux_loss_coeff / (num_tokens * num_tokens * topk)
    )
    return aux_loss

# 将 辅助损失（auxiliary loss）保存到 损失追踪器 中
def save_to_aux_losses_tracker(
    name: str,
    loss: torch.Tensor,
    layer_number: int,
    num_layers: int,
    reduce_group: torch.distributed.ProcessGroup = None,
    avg_group: torch.distributed.ProcessGroup = None,
):
    """Save the auxiliary loss for logging.
    Args:
        name (str): The name of the loss.
        loss (torch.Tensor): The loss tensor.
        layer_number (int): Layer index of the loss.
        num_layers (int): The number of total layers.
        reduce_group (torch.distributed.ProcessGroup): The group for reducing the loss.
        mean_group (torch.distributed.ProcessGroup): The group for averaging the loss.
    """
    # Skip aux loss logging if layer_number is None.
    if layer_number is None:
        return

    tracker = {}
    if name not in tracker:
        tracker[name] = {}
        tracker[name]["values"] = torch.zeros(num_layers, device=loss.device)
    tracker[name]["values"][layer_number - 1] += loss.detach()  # Aggregate the loss for the layer.
    tracker[name]["reduce_group"] = reduce_group
    tracker[name]["avg_group"] = avg_group

# 计算并应用 负载均衡辅助损失（load balancing auxiliary loss）到 MoE（Mixture of Experts）模型中的某个层
# 返回负载均衡损失以及经过负载均衡损失调整的激活张量
def apply_load_balancing_loss(
    probs: torch.Tensor,
    topk, 
    num_local_tokens_per_expert: torch.Tensor,
    activation: torch.Tensor,
):
    """Applies auxiliary loss to the MoE layer.

    Args:
        probs (torch.Tensor): The probs output by the router for each token. [num_tokens, num_experts]
        num_local_tokens_per_expert (torch.Tensor): The number of tokens per expert. [num_experts]
        activation (torch.Tensor): The activation tensor to attach the gradient function to.

    Returns:
        torch.Tensor: The activation tensor with the attached gradient function.
    """
    # 辅助损失的系数，用于缩放损失
    moe_aux_loss_coeff = 1e-3
    aux_loss = switch_load_balancing_loss_func(
        probs,
        num_local_tokens_per_expert,
        topk,
        moe_aux_loss_coeff,
        sequence_partition_group=None,
    )
    save_to_aux_losses_tracker(
        "load_balancing_loss",
        aux_loss / moe_aux_loss_coeff,
        32,
        32
    )
    activation = MoEAuxLossAutoScaler.apply(activation, aux_loss)
    return aux_loss, activation

# 在 MoE（Mixture of Experts）模型 中根据损失来进行负载均衡调整，并计算相关的 辅助损失。
def aux_loss_load_balancing(logits, topk, capacity_factor, drop_policy, pad_to_capacity):
    """Apply loss-based load balancing to the logits tensor.

    Args:
        logits (torch.Tensor): the logits tensor after gating, shape: [num_tokens, num_experts].

    Returns:
        probs (torch.Tensor): the probabilities tensor after load balancing.
        indices (torch.Tensor): the indices tensor after top-k selection.
    """
    probs, indices, tokens_per_expert, topk_masked_gates, final_mask = topk_softmax_with_capacity(
        logits,
        topk,
        capacity_factor=capacity_factor,
        drop_policy=drop_policy,
        pad_to_capacity=pad_to_capacity,
        use_pre_softmax=False,
    )

    # if self.training:
        # Apply load balancing loss
    scores = torch.softmax(logits, dim=-1, dtype=torch.float32)
    aux_loss, probs = apply_load_balancing_loss(scores, topk, tokens_per_expert, activation=probs)
    return aux_loss, probs, indices, tokens_per_expert, scores, topk_masked_gates, final_mask

def megatron_topgating(
    logits,
    topk: int,
    capacity_factor: float,
    drop_policy: str,
    pad_to_capacity: bool
):
    aux_loss, probs, indices, tokens_per_expert, scores, topk_masked_gates, final_mask  = aux_loss_load_balancing(logits, topk, capacity_factor, drop_policy, pad_to_capacity)
    return aux_loss, probs, indices, topk_masked_gates, final_mask


if __name__ == "__main__":
    logits = torch.randn(16, 8).to('cuda')
    megatron_topgating(logits, 4, 1.0, 'probs', False)

'''
常见模型deepseek系列  mixtral系列  qwen-moe系列 

输入：
sequenselen S:[2048, 4096, 8192]
expert_number E:[8, 16, 64, 128]
logits: [S, E]
topk:[2, 8, 16]  topk <  expert_number
capacity_factor: [None,  1.0]
drop_policy: [probs, position]
if capacity_factor is not None:
    pad_to_capacity:[true, false]
'''