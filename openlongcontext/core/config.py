"""
Configuration management for OpenLongContext.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import yaml
import json
from omegaconf import OmegaConf, DictConfig


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str = "longformer"
    model_type: str = "transformer"
    hidden_size: int = 768
    num_layers: int = 12
    num_attention_heads: int = 12
    max_position_embeddings: int = 4096
    attention_window: Optional[int] = 512
    dropout: float = 0.1
    activation: str = "gelu"
    vocab_size: int = 50265
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5
    gradient_checkpointing: bool = False
    use_cache: bool = True
    additional_params: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class DataConfig:
    """Data configuration."""
    dataset_name: str = "wikitext"
    dataset_config: Optional[str] = None
    data_dir: Optional[str] = None
    max_length: int = 1024
    batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = False
    shuffle: bool = True
    validation_split: float = 0.1
    test_split: float = 0.1
    preprocessing: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    """Training configuration."""
    num_epochs: int = 10
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    warmup_steps: int = 1000
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "linear"
    gradient_accumulation_steps: int = 1
    fp16: bool = False
    fp16_opt_level: str = "O1"
    local_rank: int = -1
    dataloader_drop_last: bool = False
    eval_steps: int = 500
    save_steps: int = 1000
    save_total_limit: int = 3
    seed: int = 42
    logging_steps: int = 100
    logging_first_step: bool = True
    run_name: Optional[str] = None
    disable_tqdm: bool = False
    log_level: str = "info"


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    eval_batch_size: int = 16
    metrics: List[str] = field(default_factory=lambda: ["perplexity", "accuracy"])
    save_predictions: bool = False
    compute_memory_footprint: bool = True
    compute_time: bool = True
    use_auth_token: bool = False
    trust_remote_code: bool = False


@dataclass
class ExperimentConfig:
    """Experiment configuration."""
    name: str = "default_experiment"
    description: str = ""
    tags: List[str] = field(default_factory=list)
    output_dir: str = "./outputs"
    cache_dir: str = "./cache"
    tensorboard_dir: str = "./tensorboard"
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    mlflow_tracking_uri: Optional[str] = None
    track_carbon: bool = False
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    hub_strategy: str = "every_save"


class Config:
    """Main configuration class."""
    
    def __init__(
        self,
        model: Optional[Union[ModelConfig, Dict[str, Any]]] = None,
        data: Optional[Union[DataConfig, Dict[str, Any]]] = None,
        training: Optional[Union[TrainingConfig, Dict[str, Any]]] = None,
        evaluation: Optional[Union[EvaluationConfig, Dict[str, Any]]] = None,
        experiment: Optional[Union[ExperimentConfig, Dict[str, Any]]] = None,
        **kwargs
    ):
        """Initialize configuration."""
        # Initialize model config
        if model is None:
            self.model = ModelConfig()
        elif isinstance(model, dict):
            self.model = ModelConfig(**model)
        else:
            self.model = model
            
        # Initialize data config
        if data is None:
            self.data = DataConfig()
        elif isinstance(data, dict):
            self.data = DataConfig(**data)
        else:
            self.data = data
            
        # Initialize training config
        if training is None:
            self.training = TrainingConfig()
        elif isinstance(training, dict):
            self.training = TrainingConfig(**training)
        else:
            self.training = training
            
        # Initialize evaluation config
        if evaluation is None:
            self.evaluation = EvaluationConfig()
        elif isinstance(evaluation, dict):
            self.evaluation = EvaluationConfig(**evaluation)
        else:
            self.evaluation = evaluation
            
        # Initialize experiment config
        if experiment is None:
            self.experiment = ExperimentConfig()
        elif isinstance(experiment, dict):
            self.experiment = ExperimentConfig(**experiment)
        else:
            self.experiment = experiment
            
        # Store additional parameters
        self.additional_params = kwargs
        
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Config":
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "Config":
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    @classmethod
    def from_omegaconf(cls, config: DictConfig) -> "Config":
        """Load configuration from OmegaConf DictConfig."""
        config_dict = OmegaConf.to_container(config, resolve=True)
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "model": self.model.__dict__,
            "data": self.data.__dict__,
            "training": self.training.__dict__,
            "evaluation": self.evaluation.__dict__,
            "experiment": self.experiment.__dict__,
            **self.additional_params
        }
    
    def save_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    def save_json(self, path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"Config(model={self.model.name}, data={self.data.dataset_name}, experiment={self.experiment.name})"