"""
Mathematical Utilities

Comprehensive mathematical utilities for long context models.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import logging
import math
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)


def safe_log(x: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """
    Compute log with numerical stability.
    
    Args:
        x: Input tensor
        eps: Small epsilon to avoid log(0)
        
    Returns:
        Log of input with numerical stability
    """
    return torch.log(torch.clamp(x, min=eps))


def safe_exp(x: torch.Tensor, max_val: float = 50.0) -> torch.Tensor:
    """
    Compute exp with numerical stability.
    
    Args:
        x: Input tensor
        max_val: Maximum value to clamp input
        
    Returns:
        Exp of input with numerical stability
    """
    return torch.exp(torch.clamp(x, max=max_val))


def log_sum_exp(x: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    """
    Numerically stable log-sum-exp.
    
    Args:
        x: Input tensor
        dim: Dimension to sum over
        keepdim: Whether to keep dimensions
        
    Returns:
        Log-sum-exp of input
    """
    max_val, _ = torch.max(x, dim=dim, keepdim=True)
    return max_val + torch.log(torch.sum(torch.exp(x - max_val), dim=dim, keepdim=True))


def gumbel_softmax(
    logits: torch.Tensor,
    temperature: float = 1.0,
    hard: bool = False,
    dim: int = -1
) -> torch.Tensor:
    """
    Gumbel-Softmax sampling.
    
    Args:
        logits: Input logits
        temperature: Temperature parameter
        hard: Whether to use hard sampling
        dim: Dimension to apply softmax
        
    Returns:
        Gumbel-softmax samples
    """
    # Sample Gumbel noise
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)

    # Add noise to logits and apply temperature
    y = (logits + gumbel_noise) / temperature

    # Softmax
    y_soft = torch.softmax(y, dim=dim)

    if hard:
        # Straight-through estimator
        y_hard = torch.zeros_like(y_soft)
        y_hard.scatter_(dim, torch.argmax(y_soft, dim=dim, keepdim=True), 1.0)
        y = y_hard - y_soft.detach() + y_soft
    else:
        y = y_soft

    return y


def positional_encoding(
    seq_len: int,
    d_model: int,
    max_len: int = 10000,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Generate sinusoidal positional encoding.
    
    Args:
        seq_len: Sequence length
        d_model: Model dimension
        max_len: Maximum sequence length for frequency calculation
        device: Device to create tensor on
        
    Returns:
        Positional encoding tensor [seq_len, d_model]
    """
    pe = torch.zeros(seq_len, d_model, device=device)
    position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)

    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float, device=device) *
        -(math.log(max_len) / d_model)
    )

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe


def rotary_positional_encoding(
    seq_len: int,
    dim: int,
    base: float = 10000.0,
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate rotary positional encoding (RoPE).
    
    Args:
        seq_len: Sequence length
        dim: Dimension (should be even)
        base: Base for frequency calculation
        device: Device to create tensors on
        
    Returns:
        Tuple of (cos, sin) tensors for RoPE
    """
    assert dim % 2 == 0, "Dimension must be even for RoPE"

    # Create frequency tensor
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float, device=device) / dim))

    # Create position tensor
    t = torch.arange(seq_len, dtype=torch.float, device=device)

    # Compute frequencies
    freqs = torch.outer(t, inv_freq)

    # Create cos and sin
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)

    return cos, sin


def apply_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
) -> torch.Tensor:
    """
    Apply rotary positional embedding to input tensor.
    
    Args:
        x: Input tensor [..., seq_len, dim]
        cos: Cosine values [seq_len, dim//2]
        sin: Sine values [seq_len, dim//2]
        
    Returns:
        Tensor with rotary positional embedding applied
    """
    # Split x into pairs
    x1, x2 = x[..., ::2], x[..., 1::2]

    # Apply rotation
    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x1 * sin + x2 * cos

    # Interleave back
    return torch.stack([rotated_x1, rotated_x2], dim=-1).flatten(-2)


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    scale: Optional[float] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Scaled dot-product attention.
    
    Args:
        query: Query tensor [batch, heads, seq_q, dim]
        key: Key tensor [batch, heads, seq_k, dim]
        value: Value tensor [batch, heads, seq_v, dim]
        mask: Attention mask [batch, heads, seq_q, seq_k]
        dropout_p: Dropout probability
        scale: Scale factor (default: 1/sqrt(dim))
        
    Returns:
        Tuple of (attention_output, attention_weights)
    """
    _, _, _, dim = query.shape

    if scale is None:
        scale = 1.0 / math.sqrt(dim)

    # Compute attention scores
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale

    # Apply mask if provided
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Softmax
    attn_weights = torch.softmax(scores, dim=-1)

    # Apply dropout
    if dropout_p > 0.0:
        attn_weights = torch.dropout(attn_weights, dropout_p, train=True)

    # Apply attention to values
    output = torch.matmul(attn_weights, value)

    return output, attn_weights


def compute_entropy(probs: torch.Tensor, dim: int = -1, eps: float = 1e-10) -> torch.Tensor:
    """
    Compute entropy of probability distribution.
    
    Args:
        probs: Probability tensor
        dim: Dimension to compute entropy over
        eps: Small epsilon for numerical stability
        
    Returns:
        Entropy tensor
    """
    log_probs = safe_log(probs, eps)
    return -torch.sum(probs * log_probs, dim=dim)


def compute_kl_divergence(
    p: torch.Tensor,
    q: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-10
) -> torch.Tensor:
    """
    Compute KL divergence between two probability distributions.
    
    Args:
        p: First probability distribution
        q: Second probability distribution
        dim: Dimension to compute KL over
        eps: Small epsilon for numerical stability
        
    Returns:
        KL divergence tensor
    """
    log_p = safe_log(p, eps)
    log_q = safe_log(q, eps)
    return torch.sum(p * (log_p - log_q), dim=dim)


def compute_js_divergence(
    p: torch.Tensor,
    q: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-10
) -> torch.Tensor:
    """
    Compute Jensen-Shannon divergence between two probability distributions.
    
    Args:
        p: First probability distribution
        q: Second probability distribution
        dim: Dimension to compute JS over
        eps: Small epsilon for numerical stability
        
    Returns:
        JS divergence tensor
    """
    m = 0.5 * (p + q)
    return 0.5 * compute_kl_divergence(p, m, dim, eps) + 0.5 * compute_kl_divergence(q, m, dim, eps)


def gaussian_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """
    Compute Gaussian (RBF) kernel between two tensors.
    
    Args:
        x: First tensor
        y: Second tensor
        sigma: Bandwidth parameter
        
    Returns:
        Kernel values
    """
    dist_sq = torch.sum((x.unsqueeze(-2) - y.unsqueeze(-3)) ** 2, dim=-1)
    return torch.exp(-dist_sq / (2 * sigma ** 2))


def polynomial_kernel(
    x: torch.Tensor,
    y: torch.Tensor,
    degree: int = 3,
    coef0: float = 1.0
) -> torch.Tensor:
    """
    Compute polynomial kernel between two tensors.
    
    Args:
        x: First tensor
        y: Second tensor
        degree: Polynomial degree
        coef0: Constant coefficient
        
    Returns:
        Kernel values
    """
    dot_product = torch.matmul(x, y.transpose(-2, -1))
    return (dot_product + coef0) ** degree


def cosine_similarity(x: torch.Tensor, y: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute cosine similarity between tensors.
    
    Args:
        x: First tensor
        y: Second tensor
        dim: Dimension to compute similarity over
        eps: Small epsilon for numerical stability
        
    Returns:
        Cosine similarity tensor
    """
    dot_product = torch.sum(x * y, dim=dim)
    norm_x = torch.norm(x, dim=dim) + eps
    norm_y = torch.norm(y, dim=dim) + eps
    return dot_product / (norm_x * norm_y)


def pairwise_distances(x: torch.Tensor, y: torch.Tensor, p: float = 2.0) -> torch.Tensor:
    """
    Compute pairwise distances between two sets of vectors.
    
    Args:
        x: First set of vectors [batch, n, dim]
        y: Second set of vectors [batch, m, dim]
        p: p-norm to use (default: 2.0 for Euclidean)
        
    Returns:
        Pairwise distance matrix [batch, n, m]
    """
    diff = x.unsqueeze(-2) - y.unsqueeze(-3)  # [batch, n, m, dim]
    return torch.norm(diff, p=p, dim=-1)


def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float('inf')
) -> torch.Tensor:
    """
    Filter logits using top-k and/or nucleus (top-p) filtering.
    
    Args:
        logits: Input logits
        top_k: Number of top tokens to keep (0 = no filtering)
        top_p: Cumulative probability threshold (1.0 = no filtering)
        filter_value: Value to set filtered logits to
        
    Returns:
        Filtered logits
    """
    if top_k > 0:
        # Top-k filtering
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        # Nucleus filtering
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value

    return logits


def compute_gradient_norm(parameters, norm_type: float = 2.0) -> float:
    """
    Compute gradient norm of model parameters.
    
    Args:
        parameters: Model parameters
        norm_type: Type of norm to compute
        
    Returns:
        Gradient norm value
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    parameters = [p for p in parameters if p.grad is not None]

    if len(parameters) == 0:
        return 0.0

    total_norm = 0.0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type

    return total_norm ** (1.0 / norm_type)


def clip_gradients(parameters, max_norm: float, norm_type: float = 2.0) -> float:
    """
    Clip gradients by norm.
    
    Args:
        parameters: Model parameters
        max_norm: Maximum norm value
        norm_type: Type of norm to use
        
    Returns:
        Total norm before clipping
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    parameters = [p for p in parameters if p.grad is not None]

    if len(parameters) == 0:
        return 0.0

    max_norm = float(max_norm)
    norm_type = float(norm_type)

    total_norm = 0.0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type

    total_norm = total_norm ** (1.0 / norm_type)

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.data.mul_(clip_coef)

    return total_norm
