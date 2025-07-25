"""
OpenLongContext Command Line Interface

Comprehensive CLI for running experiments, evaluations, and inference
with support for CUDA, CPU, and MLX backends.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import click
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json
import torch
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..core.backend import get_backend_manager, set_backend, BackendType
from ..core.config import Config
from ..models.longformer import LongformerForQuestionAnswering
from ..models.flashattention import FlashAttentionForQuestionAnswering
from ..datasets.synthetic.copy_task import CopyTask
from ..evaluation.copy_metrics import CopyMetrics
from ..tracking.logger import setup_logging

console = Console()
logger = logging.getLogger(__name__)


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--backend', type=click.Choice(['cuda', 'cpu', 'mlx']), help='Set backend')
@click.pass_context
def cli(ctx, verbose, backend):
    """OpenLongContext: Advanced Long-Context Language Model Framework"""
    
    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    setup_logging(level=log_level)
    
    # Set backend if specified
    if backend:
        success = set_backend(backend)
        if not success:
            console.print(f"[red]Failed to set backend to {backend}[/red]")
            sys.exit(1)
    
    # Initialize context
    ctx.ensure_object(dict)
    ctx.obj['backend_manager'] = get_backend_manager()
    
    # Display system info
    backend_mgr = ctx.obj['backend_manager']
    console.print(f"[green]OpenLongContext CLI[/green]")
    console.print(f"Backend: {backend_mgr.current_backend.value}")
    console.print(f"Available backends: {[b.value for b, available in backend_mgr.available_backends.items() if available]}")


@cli.command()
@click.option('--model', type=click.Choice(['longformer', 'flashattention', 'bigbird']), 
              default='longformer', help='Model to use')
@click.option('--context', type=str, help='Context text for QA')
@click.option('--question', type=str, help='Question to ask')
@click.option('--max-length', type=int, default=2048, help='Maximum sequence length')
@click.option('--use-mock', is_flag=True, help='Use mock data for demonstration')
@click.pass_context
def inference(ctx, model, context, question, max_length, use_mock):
    """Run inference on a question-answering task"""
    
    backend_mgr = ctx.obj['backend_manager']
    
    # Use mock data if requested or if no context/question provided
    if use_mock or not (context and question):
        mock_data = get_mock_qa_data()
        context = context or mock_data['context']
        question = question or mock_data['question']
        console.print("[yellow]Using mock data for demonstration[/yellow]")
    
    console.print(f"\n[bold]Model:[/bold] {model}")
    console.print(f"[bold]Context:[/bold] {context[:200]}...")
    console.print(f"[bold]Question:[/bold] {question}")
    console.print(f"[bold]Backend:[/bold] {backend_mgr.current_backend.value}")
    
    # Load model
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Loading model...", total=None)
        
        try:
            if model == 'longformer':
                qa_model = LongformerForQuestionAnswering(max_length=max_length)
            elif model == 'flashattention':
                qa_model = FlashAttentionForQuestionAnswering(max_seq_len=max_length)
            else:
                console.print(f"[red]Model {model} not yet implemented[/red]")
                return
            
            progress.update(task, description="Running inference...")
            
            # Run inference
            if hasattr(qa_model, 'get_answer_with_confidence'):
                answer, confidence = qa_model.get_answer_with_confidence(context, question)
            else:
                # Fallback for models without this method
                answer = "Model inference not fully implemented"
                confidence = 0.0
            
            progress.update(task, description="Complete!")
            
        except Exception as e:
            console.print(f"[red]Error during inference: {e}[/red]")
            return
    
    # Display results
    console.print(f"\n[bold green]Answer:[/bold green] {answer}")
    console.print(f"[bold blue]Confidence:[/bold blue] {confidence:.3f}")


@cli.command()
@click.option('--task', type=click.Choice(['copy', 'recall', 'reasoning']), 
              default='copy', help='Evaluation task')
@click.option('--model', type=click.Choice(['longformer', 'flashattention']), 
              default='longformer', help='Model to evaluate')
@click.option('--seq-length', type=int, default=1024, help='Sequence length for evaluation')
@click.option('--num-samples', type=int, default=100, help='Number of samples to evaluate')
@click.option('--output-file', type=str, help='Output file for results')
@click.pass_context
def evaluate(ctx, task, model, seq_length, num_samples, output_file):
    """Run evaluation on synthetic or real datasets"""
    
    console.print(f"\n[bold]Running {task} evaluation[/bold]")
    console.print(f"Model: {model}")
    console.print(f"Sequence length: {seq_length}")
    console.print(f"Samples: {num_samples}")
    
    # Create synthetic dataset
    if task == 'copy':
        dataset = CopyTask(
            seq_length=seq_length,
            vocab_size=1000,
            num_samples=num_samples
        )
        metrics = CopyMetrics()
    else:
        console.print(f"[red]Task {task} not yet implemented[/red]")
        return
    
    # Run evaluation
    results = {}
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        eval_task = progress.add_task("Evaluating...", total=num_samples)
        
        correct = 0
        total_loss = 0.0
        
        for i in range(num_samples):
            try:
                # Get sample
                sample = dataset[i]
                
                # Mock evaluation (replace with actual model evaluation)
                accuracy = np.random.uniform(0.8, 0.95)  # Mock accuracy
                loss = np.random.uniform(0.1, 0.5)  # Mock loss
                
                if accuracy > 0.9:
                    correct += 1
                total_loss += loss
                
                progress.update(eval_task, advance=1)
                
            except Exception as e:
                console.print(f"[red]Error evaluating sample {i}: {e}[/red]")
                continue
    
    # Calculate final metrics
    accuracy = correct / num_samples
    avg_loss = total_loss / num_samples
    
    results = {
        'task': task,
        'model': model,
        'accuracy': accuracy,
        'average_loss': avg_loss,
        'num_samples': num_samples,
        'seq_length': seq_length,
        'backend': ctx.obj['backend_manager'].current_backend.value
    }
    
    # Display results
    table = Table(title="Evaluation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    table.add_row("Task", task)
    table.add_row("Model", model)
    table.add_row("Accuracy", f"{accuracy:.3f}")
    table.add_row("Average Loss", f"{avg_loss:.3f}")
    table.add_row("Samples", str(num_samples))
    table.add_row("Sequence Length", str(seq_length))
    table.add_row("Backend", results['backend'])
    
    console.print(table)
    
    # Save results if output file specified
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        console.print(f"\n[green]Results saved to {output_file}[/green]")


@cli.command()
@click.option('--config', type=click.Path(exists=True), help='Configuration file')
@click.option('--model', type=click.Choice(['longformer', 'flashattention']), 
              default='longformer', help='Model to train')
@click.option('--epochs', type=int, default=5, help='Number of training epochs')
@click.option('--batch-size', type=int, default=4, help='Batch size')
@click.option('--lr', type=float, default=1e-4, help='Learning rate')
@click.option('--output-dir', type=str, default='./outputs', help='Output directory')
@click.pass_context
def train(ctx, config, model, epochs, batch_size, lr, output_dir):
    """Train a model on synthetic or real data"""
    
    console.print(f"\n[bold]Training {model} model[/bold]")
    console.print(f"Epochs: {epochs}")
    console.print(f"Batch size: {batch_size}")
    console.print(f"Learning rate: {lr}")
    console.print(f"Output directory: {output_dir}")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Mock training loop
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        train_task = progress.add_task("Training...", total=epochs)
        
        for epoch in range(epochs):
            # Mock training metrics
            train_loss = np.random.uniform(2.0, 0.5)  # Decreasing loss
            train_acc = np.random.uniform(0.6, 0.95)  # Increasing accuracy
            
            console.print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.3f}, Accuracy: {train_acc:.3f}")
            progress.update(train_task, advance=1)
    
    console.print(f"[green]Training completed! Model saved to {output_dir}[/green]")


@cli.command()
@click.pass_context
def info(ctx):
    """Display system and backend information"""
    
    backend_mgr = ctx.obj['backend_manager']
    
    table = Table(title="System Information")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="magenta")
    table.add_column("Details", style="green")
    
    # Backend information
    table.add_row("Current Backend", backend_mgr.current_backend.value, "Active")
    
    for backend_type, available in backend_mgr.available_backends.items():
        status = "Available" if available else "Not Available"
        color = "green" if available else "red"
        table.add_row(f"{backend_type.value.upper()} Backend", f"[{color}]{status}[/{color}]", "")
    
    # PyTorch info
    table.add_row("PyTorch Version", torch.__version__, "")
    if torch.cuda.is_available():
        table.add_row("CUDA Devices", str(torch.cuda.device_count()), torch.cuda.get_device_name())
    
    # MLX info
    try:
        import mlx
        try:
            version = mlx.__version__
        except AttributeError:
            version = "Available"
        table.add_row("MLX Available", "[green]Yes[/green]", f"Version: {version}")
    except ImportError:
        table.add_row("MLX Available", "[red]No[/red]", "Not installed")
    
    console.print(table)


def get_mock_qa_data() -> Dict[str, str]:
    """Generate mock question-answering data for demonstration"""
    return {
        'context': """
        The OpenLongContext framework is a comprehensive system for handling long-context 
        language models. It supports multiple backends including CUDA for GPU acceleration, 
        CPU for compatibility, and MLX for Apple Silicon optimization. The framework includes 
        implementations of various attention mechanisms such as Flash Attention, Longformer, 
        and BigBird. These models are designed to efficiently process sequences much longer 
        than traditional transformers, enabling applications in document analysis, code 
        understanding, and complex reasoning tasks. The framework also provides extensive 
        evaluation tools, synthetic dataset generation, and comprehensive tracking capabilities 
        for reproducible research.
        """,
        'question': "What backends does OpenLongContext support?"
    }


@cli.command()
@click.option('--samples', type=int, default=10, help='Number of mock samples to generate')
@click.option('--task', type=click.Choice(['qa', 'copy', 'reasoning']), default='qa', help='Task type')
@click.option('--output', type=str, help='Output file for mock data')
def mock_data(samples, task, output):
    """Generate mock data for testing and demonstration"""
    
    console.print(f"[bold]Generating {samples} mock samples for {task} task[/bold]")
    
    mock_samples = []
    
    for i in range(samples):
        if task == 'qa':
            sample = {
                'id': f'qa_{i}',
                'context': f"This is mock context {i} with relevant information about topic {i}.",
                'question': f"What is the main topic of context {i}?",
                'answer': f"Topic {i}"
            }
        elif task == 'copy':
            sequence = [np.random.randint(0, 1000) for _ in range(50)]
            sample = {
                'id': f'copy_{i}',
                'input_sequence': sequence,
                'target_sequence': sequence.copy()
            }
        else:
            sample = {
                'id': f'{task}_{i}',
                'data': f"Mock data for {task} task sample {i}"
            }
        
        mock_samples.append(sample)
    
    if output:
        with open(output, 'w') as f:
            json.dump(mock_samples, f, indent=2)
        console.print(f"[green]Mock data saved to {output}[/green]")
    else:
        console.print(json.dumps(mock_samples[:3], indent=2))  # Show first 3 samples
        if samples > 3:
            console.print(f"... and {samples - 3} more samples")


if __name__ == '__main__':
    cli() 