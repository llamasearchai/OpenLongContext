"""
Backend Management System for OpenLongContext

Provides unified interface for CUDA, CPU, and MLX (Apple Silicon) backends
with automatic detection and graceful fallbacks.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import logging
from enum import Enum
from typing import Dict, Union

logger = logging.getLogger(__name__)


class BackendType(Enum):
    """Supported backend types."""
    CUDA = "cuda"
    CPU = "cpu"
    MLX = "mlx"


class BackendManager:
    """
    Manages backend selection and provides unified interface across
    CUDA, CPU, and MLX (Apple Silicon) backends.
    """

    def __init__(self):
        self._available_backends = self._detect_backends()
        self._current_backend = self._select_default_backend()

    def _detect_backends(self) -> Dict[BackendType, bool]:
        """Detect available backends on the current system."""
        backends = {
            BackendType.CPU: True,  # Always available
            BackendType.CUDA: False,
            BackendType.MLX: False
        }

        # Check CUDA availability
        try:
            import torch
            if torch.cuda.is_available():
                backends[BackendType.CUDA] = True
                logger.info(f"CUDA detected: {torch.cuda.device_count()} devices")
        except ImportError:
            logger.debug("PyTorch not available, CUDA backend disabled")

        # Check MLX availability (Apple Silicon)
        try:
            import mlx.core  # noqa: F401
            backends[BackendType.MLX] = True
            logger.info("MLX backend available (Apple Silicon)")
        except ImportError:
            logger.debug("MLX not available, backend disabled")

        return backends

    def _select_default_backend(self) -> BackendType:
        """Select the best available backend by priority."""
        if self._available_backends[BackendType.CUDA]:
            return BackendType.CUDA
        elif self._available_backends[BackendType.MLX]:
            return BackendType.MLX
        else:
            return BackendType.CPU

    @property
    def current_backend(self) -> BackendType:
        """Get the currently selected backend."""
        return self._current_backend

    @property
    def available_backends(self) -> Dict[BackendType, bool]:
        """Get dictionary of available backends."""
        return self._available_backends.copy()

    def set_backend(self, backend: Union[str, BackendType]) -> bool:
        """
        Set the active backend.
        
        Args:
            backend: Backend to set ('cuda', 'cpu', 'mlx' or BackendType enum)
            
        Returns:
            True if backend was set successfully, False otherwise
        """
        if isinstance(backend, str):
            try:
                backend = BackendType(backend.lower())
            except ValueError:
                logger.error(f"Invalid backend: {backend}")
                return False

        if not self._available_backends[backend]:
            logger.warning(f"Backend {backend.value} not available, keeping {self._current_backend.value}")
            return False

        self._current_backend = backend
        logger.info(f"Backend set to: {backend.value}")
        return True

    def get_device_string(self) -> str:
        """Get device string for the current backend."""
        if self._current_backend == BackendType.CUDA:
            return "cuda"
        elif self._current_backend == BackendType.MLX:
            return "mlx"
        else:
            return "cpu"

    def get_tensor_library(self):
        """Get the appropriate tensor library for current backend."""
        if self._current_backend == BackendType.CUDA or self._current_backend == BackendType.CPU:
            import torch
            return torch
        elif self._current_backend == BackendType.MLX:
            try:
                import mlx.core as mx
                return mx
            except ImportError:
                logger.warning("MLX not available, falling back to PyTorch")
                import torch
                return torch
        else:
            import torch
            return torch

    def create_tensor(self, data, dtype=None, device=None):
        """Create tensor using appropriate backend."""
        tensor_lib = self.get_tensor_library()

        if self._current_backend == BackendType.MLX:
            # MLX tensor creation
            import mlx.core as mx
            if dtype is None:
                return mx.array(data)
            else:
                return mx.array(data, dtype=dtype)
        else:
            # PyTorch tensor creation
            import torch
            device = device or self.get_device_string()
            if dtype is None:
                return torch.tensor(data, device=device)
            else:
                return torch.tensor(data, dtype=dtype, device=device)

    def is_cuda_available(self) -> bool:
        """Check if CUDA backend is available."""
        return self._available_backends[BackendType.CUDA]

    def is_mlx_available(self) -> bool:
        """Check if MLX backend is available."""
        return self._available_backends[BackendType.MLX]

    def get_flash_attention(self):
        """Get flash attention implementation if available."""
        if self._current_backend == BackendType.CUDA and self.is_cuda_available():
            try:
                import flash_attn
                return flash_attn
            except ImportError:
                logger.warning("flash-attn not available, using standard attention")
                return None
        return None


# Global backend manager instance
backend_manager = BackendManager()


def get_backend_manager() -> BackendManager:
    """Get the global backend manager instance."""
    return backend_manager


def set_backend(backend: Union[str, BackendType]) -> bool:
    """Set the global backend."""
    return backend_manager.set_backend(backend)


def get_current_backend() -> BackendType:
    """Get the current backend."""
    return backend_manager.current_backend


def get_device() -> str:
    """Get the current device string."""
    return backend_manager.get_device_string()


def create_tensor(data, dtype=None, device=None):
    """Create tensor using current backend."""
    return backend_manager.create_tensor(data, dtype, device)
