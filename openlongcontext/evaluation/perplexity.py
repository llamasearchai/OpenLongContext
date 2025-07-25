"""
Perplexity Evaluation

Comprehensive perplexity evaluation metrics for language models.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import logging
from collections import defaultdict
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

logger = logging.getLogger(__name__)


def compute_perplexity(
    logits: torch.Tensor,
    targets: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    reduction: str = "mean"
) -> float:
    """
    Compute perplexity from model logits and targets.
    
    Args:
        logits: Model logits [batch_size, seq_len, vocab_size]
        targets: Target token IDs [batch_size, seq_len]
        attention_mask: Optional attention mask [batch_size, seq_len]
        reduction: How to reduce across batch ("mean", "sum", "none")
        
    Returns:
        Perplexity value
    """
    # Flatten logits and targets
    logits_flat = logits.view(-1, logits.size(-1))
    targets_flat = targets.view(-1)

    # Compute cross-entropy loss
    loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')
    loss = loss.view(targets.shape)  # Reshape back to [batch_size, seq_len]

    # Apply attention mask if provided
    if attention_mask is not None:
        loss = loss * attention_mask
        if reduction == "mean":
            loss = loss.sum() / attention_mask.sum()
        elif reduction == "sum":
            loss = loss.sum()
    else:
        if reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()

    # Convert to perplexity
    if reduction == "none":
        return torch.exp(loss)
    else:
        return torch.exp(loss).item()


def compute_token_level_perplexity(
    logits: torch.Tensor,
    targets: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute token-level perplexity.
    
    Args:
        logits: Model logits [batch_size, seq_len, vocab_size]
        targets: Target token IDs [batch_size, seq_len]
        attention_mask: Optional attention mask [batch_size, seq_len]
        
    Returns:
        Token-level perplexity [batch_size, seq_len]
    """
    # Compute token-level cross-entropy
    logits_flat = logits.view(-1, logits.size(-1))
    targets_flat = targets.view(-1)

    loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')
    loss = loss.view(targets.shape)

    # Apply attention mask
    if attention_mask is not None:
        loss = loss * attention_mask

    # Convert to perplexity
    return torch.exp(loss)


def evaluate_model_perplexity(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    max_length: Optional[int] = None,
    stride: Optional[int] = None
) -> Dict[str, any]:
    """
    Evaluate model perplexity on a dataset.
    
    Args:
        model: Language model to evaluate
        dataloader: DataLoader containing evaluation data
        device: Device to run evaluation on
        max_length: Maximum sequence length for sliding window evaluation
        stride: Stride for sliding window evaluation
        
    Returns:
        Dictionary containing perplexity metrics
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    perplexities = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating perplexity"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            if max_length is not None and input_ids.size(1) > max_length:
                # Use sliding window evaluation
                batch_perplexity = evaluate_sliding_window_perplexity(
                    model, input_ids, attention_mask, max_length, stride or max_length // 2
                )
                perplexities.extend(batch_perplexity)
            else:
                # Standard evaluation
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs

                # Shift logits and targets for next-token prediction
                shift_logits = logits[..., :-1, :].contiguous()
                shift_targets = input_ids[..., 1:].contiguous()
                shift_mask = attention_mask[..., 1:] if attention_mask is not None else None

                # Compute loss
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_targets.view(-1),
                    reduction='none'
                )
                loss = loss.view(shift_targets.shape)

                if shift_mask is not None:
                    loss = loss * shift_mask
                    batch_tokens = shift_mask.sum().item()
                    batch_loss = loss.sum().item()
                else:
                    batch_tokens = shift_targets.numel()
                    batch_loss = loss.sum().item()

                total_loss += batch_loss
                total_tokens += batch_tokens

                # Compute batch perplexity
                if batch_tokens > 0:
                    batch_perplexity = np.exp(batch_loss / batch_tokens)
                    perplexities.append(batch_perplexity)

    # Compute overall metrics
    overall_perplexity = np.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')

    return {
        "perplexity": overall_perplexity,
        "loss": total_loss / total_tokens if total_tokens > 0 else float('inf'),
        "total_tokens": total_tokens,
        "batch_perplexities": perplexities,
        "mean_batch_perplexity": np.mean(perplexities) if perplexities else float('inf'),
        "std_batch_perplexity": np.std(perplexities) if perplexities else 0.0
    }


def evaluate_sliding_window_perplexity(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    max_length: int,
    stride: int
) -> List[float]:
    """
    Evaluate perplexity using sliding window approach for long sequences.
    
    Args:
        model: Language model
        input_ids: Input token IDs [batch_size, seq_len]
        attention_mask: Attention mask [batch_size, seq_len]
        max_length: Maximum window length
        stride: Stride between windows
        
    Returns:
        List of perplexity values for each window
    """
    _, seq_len = input_ids.shape
    perplexities = []

    for i in range(0, seq_len - max_length + 1, stride):
        end_pos = min(i + max_length, seq_len)

        # Extract window
        window_input = input_ids[:, i:end_pos]
        window_mask = attention_mask[:, i:end_pos] if attention_mask is not None else None

        # Forward pass
        outputs = model(window_input, attention_mask=window_mask)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs

        # Compute perplexity for this window
        shift_logits = logits[..., :-1, :].contiguous()
        shift_targets = window_input[..., 1:].contiguous()
        shift_mask = window_mask[..., 1:] if window_mask is not None else None

        window_perplexity = compute_perplexity(shift_logits, shift_targets, shift_mask)
        perplexities.append(window_perplexity)

    return perplexities


def compute_conditional_perplexity(
    model: torch.nn.Module,
    context: torch.Tensor,
    target: torch.Tensor,
    device: torch.device
) -> float:
    """
    Compute conditional perplexity P(target | context).
    
    Args:
        model: Language model
        context: Context tokens [batch_size, context_len]
        target: Target tokens [batch_size, target_len]
        device: Device to run on
        
    Returns:
        Conditional perplexity
    """
    model.eval()

    # Concatenate context and target
    full_input = torch.cat([context, target], dim=1).to(device)
    context_len = context.size(1)

    with torch.no_grad():
        outputs = model(full_input)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs

        # Only compute loss on target tokens
        target_logits = logits[:, context_len-1:-1, :]  # Shift for next-token prediction
        target_tokens = target

        perplexity = compute_perplexity(target_logits, target_tokens)
        return perplexity


def analyze_perplexity_by_position(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    max_positions: int = 1024
) -> Dict[str, np.ndarray]:
    """
    Analyze how perplexity varies by token position.
    
    Args:
        model: Language model
        dataloader: DataLoader containing evaluation data
        device: Device to run on
        max_positions: Maximum number of positions to analyze
        
    Returns:
        Dictionary containing position-wise perplexity analysis
    """
    model.eval()
    position_losses = defaultdict(list)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Analyzing position-wise perplexity"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            # Limit sequence length
            seq_len = min(input_ids.size(1), max_positions + 1)
            input_ids = input_ids[:, :seq_len]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :seq_len]

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs

            # Compute token-level perplexity
            shift_logits = logits[..., :-1, :].contiguous()
            shift_targets = input_ids[..., 1:].contiguous()
            shift_mask = attention_mask[..., 1:] if attention_mask is not None else None

            token_perplexities = compute_token_level_perplexity(
                shift_logits, shift_targets, shift_mask
            )

            # Collect perplexities by position
            for pos in range(token_perplexities.size(1)):
                if shift_mask is None or shift_mask[:, pos].sum() > 0:
                    valid_perplexities = token_perplexities[:, pos]
                    if shift_mask is not None:
                        valid_perplexities = valid_perplexities[shift_mask[:, pos].bool()]
                    position_losses[pos].extend(valid_perplexities.cpu().numpy())

    # Compute statistics for each position
    positions = sorted(position_losses.keys())
    mean_perplexities = []
    std_perplexities = []

    for pos in positions:
        perplexities = np.array(position_losses[pos])
        mean_perplexities.append(np.mean(perplexities))
        std_perplexities.append(np.std(perplexities))

    return {
        "positions": np.array(positions),
        "mean_perplexity": np.array(mean_perplexities),
        "std_perplexity": np.array(std_perplexities),
        "raw_data": dict(position_losses)
    }


def plot_perplexity_analysis(
    position_analysis: Dict[str, np.ndarray],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot perplexity analysis results.
    
    Args:
        position_analysis: Results from analyze_perplexity_by_position
        save_path: Optional path to save the plot
        
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    positions = position_analysis["positions"]
    mean_perplexity = position_analysis["mean_perplexity"]
    std_perplexity = position_analysis["std_perplexity"]

    # Plot mean perplexity by position
    ax1.plot(positions, mean_perplexity, 'b-', linewidth=2, label='Mean Perplexity')
    ax1.fill_between(
        positions,
        mean_perplexity - std_perplexity,
        mean_perplexity + std_perplexity,
        alpha=0.3,
        color='blue',
        label='±1 Std Dev'
    )
    ax1.set_xlabel('Token Position')
    ax1.set_ylabel('Perplexity')
    ax1.set_title('Perplexity by Token Position')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot log perplexity for better visualization
    ax2.plot(positions, np.log(mean_perplexity), 'r-', linewidth=2, label='Log Mean Perplexity')
    ax2.fill_between(
        positions,
        np.log(mean_perplexity - std_perplexity),
        np.log(mean_perplexity + std_perplexity),
        alpha=0.3,
        color='red',
        label='±1 Std Dev (log scale)'
    )
    ax2.set_xlabel('Token Position')
    ax2.set_ylabel('Log Perplexity')
    ax2.set_title('Log Perplexity by Token Position')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def compare_model_perplexities(
    models: Dict[str, torch.nn.Module],
    dataloader: torch.utils.data.DataLoader,
    device: torch.device
) -> Dict[str, Dict[str, float]]:
    """
    Compare perplexity across multiple models.
    
    Args:
        models: Dictionary mapping model names to models
        dataloader: DataLoader containing evaluation data
        device: Device to run evaluation on
        
    Returns:
        Dictionary containing perplexity results for each model
    """
    results = {}

    for model_name, model in models.items():
        logger.info(f"Evaluating perplexity for {model_name}")
        model_results = evaluate_model_perplexity(model, dataloader, device)
        results[model_name] = model_results
        logger.info(f"{model_name} perplexity: {model_results['perplexity']:.2f}")

    return results


def compute_bits_per_byte(perplexity: float, vocab_size: int) -> float:
    """
    Convert perplexity to bits per byte (approximate).
    
    Args:
        perplexity: Perplexity value
        vocab_size: Vocabulary size
        
    Returns:
        Bits per byte
    """
    # Approximate conversion assuming average token represents ~4 characters
    bits_per_token = np.log2(perplexity)
    bits_per_byte = bits_per_token / 4.0  # Rough approximation
    return bits_per_byte
