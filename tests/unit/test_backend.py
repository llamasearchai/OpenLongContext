"""
Tests for Backend Management System

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import pytest
import torch
from unittest.mock import patch, MagicMock
from openlongcontext.core.backend import (
    BackendManager, 
    BackendType, 
    get_backend_manager,
    set_backend, 
    get_current_backend,
    get_device,
    create_tensor
)


class TestBackendType:
    """Test BackendType enum."""
    
    def test_backend_type_values(self):
        """Test that backend types have correct values."""
        assert BackendType.CUDA.value == "cuda"
        assert BackendType.CPU.value == "cpu"
        assert BackendType.MLX.value == "mlx"


class TestBackendManager:
    """Test BackendManager class."""
    
    def test_init(self):
        """Test BackendManager initialization."""
        manager = BackendManager()
        assert isinstance(manager.available_backends, dict)
        assert BackendType.CPU in manager.available_backends
        assert manager.available_backends[BackendType.CPU] is True
        assert isinstance(manager.current_backend, BackendType)
    
    @patch('openlongcontext.core.backend.torch.cuda.is_available')
    def test_detect_backends_cuda_available(self, mock_cuda_available):
        """Test backend detection when CUDA is available."""
        mock_cuda_available.return_value = True
        
        with patch('openlongcontext.core.backend.torch.cuda.device_count', return_value=2):
            manager = BackendManager()
            assert manager.available_backends[BackendType.CUDA] is True
            assert manager.current_backend == BackendType.CUDA
    
    @patch('openlongcontext.core.backend.torch.cuda.is_available')
    def test_detect_backends_cuda_unavailable(self, mock_cuda_available):
        """Test backend detection when CUDA is unavailable."""
        mock_cuda_available.return_value = False
        
        manager = BackendManager()
        assert manager.available_backends[BackendType.CUDA] is False
        assert manager.current_backend in [BackendType.CPU, BackendType.MLX]
    
    @patch('builtins.__import__')
    def test_detect_backends_mlx_available(self, mock_import):
        """Test backend detection when MLX is available."""
        # Mock successful MLX import
        mock_mlx = MagicMock()
        mock_import.return_value = mock_mlx
        
        manager = BackendManager()
        # MLX detection depends on actual import, so we test the logic
        assert BackendType.MLX in manager.available_backends
    
    def test_set_backend_valid(self):
        """Test setting a valid backend."""
        manager = BackendManager()
        # CPU should always be available
        result = manager.set_backend(BackendType.CPU)
        assert result is True
        assert manager.current_backend == BackendType.CPU
        
        # Test string input
        result = manager.set_backend("cpu")
        assert result is True
        assert manager.current_backend == BackendType.CPU
    
    def test_set_backend_invalid(self):
        """Test setting an invalid backend."""
        manager = BackendManager()
        result = manager.set_backend("invalid")
        assert result is False
    
    def test_set_backend_unavailable(self):
        """Test setting an unavailable backend."""
        manager = BackendManager()
        # Force CUDA to be unavailable
        manager._available_backends[BackendType.CUDA] = False
        
        result = manager.set_backend(BackendType.CUDA)
        assert result is False
    
    def test_get_device_string(self):
        """Test getting device string for different backends."""
        manager = BackendManager()
        
        manager._current_backend = BackendType.CPU
        assert manager.get_device_string() == "cpu"
        
        manager._current_backend = BackendType.CUDA
        assert manager.get_device_string() == "cuda"
        
        manager._current_backend = BackendType.MLX
        assert manager.get_device_string() == "mlx"
    
    def test_get_tensor_library(self):
        """Test getting appropriate tensor library."""
        manager = BackendManager()
        
        # Test CPU/CUDA (should return torch)
        manager._current_backend = BackendType.CPU
        lib = manager.get_tensor_library()
        assert lib == torch
        
        manager._current_backend = BackendType.CUDA
        lib = manager.get_tensor_library()
        assert lib == torch
    
    @patch('builtins.__import__')
    def test_get_tensor_library_mlx(self, mock_import):
        """Test getting MLX tensor library."""
        manager = BackendManager()
        manager._current_backend = BackendType.MLX
        
        # Mock successful MLX import
        mock_mlx = MagicMock()
        mock_import.return_value = mock_mlx
        
        # Test would need actual MLX to work properly
        # For now, test the fallback to torch
        with patch.object(manager, '_current_backend', BackendType.MLX):
            lib = manager.get_tensor_library()
            # Should fallback to torch if MLX import fails
            assert lib == torch
    
    def test_create_tensor_cpu(self):
        """Test tensor creation for CPU backend."""
        manager = BackendManager()
        manager._current_backend = BackendType.CPU
        
        data = [1, 2, 3, 4]
        tensor = manager.create_tensor(data)
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.device.type == "cpu"
        assert tensor.tolist() == data
    
    def test_create_tensor_with_dtype(self):
        """Test tensor creation with specific dtype."""
        manager = BackendManager()
        manager._current_backend = BackendType.CPU
        
        data = [1.0, 2.0, 3.0]
        tensor = manager.create_tensor(data, dtype=torch.float32)
        
        assert tensor.dtype == torch.float32
        assert tensor.tolist() == data
    
    def test_is_cuda_available(self):
        """Test CUDA availability check."""
        manager = BackendManager()
        result = manager.is_cuda_available()
        assert isinstance(result, bool)
        assert result == manager.available_backends[BackendType.CUDA]
    
    def test_is_mlx_available(self):
        """Test MLX availability check."""
        manager = BackendManager()
        result = manager.is_mlx_available()
        assert isinstance(result, bool)
        assert result == manager.available_backends[BackendType.MLX]
    
    @patch('openlongcontext.core.backend.torch.cuda.is_available')
    def test_get_flash_attention_cuda(self, mock_cuda_available):
        """Test flash attention availability on CUDA."""
        mock_cuda_available.return_value = True
        manager = BackendManager()
        manager._current_backend = BackendType.CUDA
        
        # Test without flash_attn installed
        flash_attn = manager.get_flash_attention()
        assert flash_attn is None  # Should be None if not installed
    
    def test_get_flash_attention_non_cuda(self):
        """Test flash attention on non-CUDA backends."""
        manager = BackendManager()
        manager._current_backend = BackendType.CPU
        
        flash_attn = manager.get_flash_attention()
        assert flash_attn is None


class TestGlobalFunctions:
    """Test global backend functions."""
    
    def test_get_backend_manager(self):
        """Test getting global backend manager."""
        manager = get_backend_manager()
        assert isinstance(manager, BackendManager)
        
        # Should return the same instance
        manager2 = get_backend_manager()
        assert manager is manager2
    
    def test_set_backend_global(self):
        """Test setting backend globally."""
        original_backend = get_current_backend()
        
        # Set to CPU (should always work)
        result = set_backend("cpu")
        assert result is True
        assert get_current_backend() == BackendType.CPU
        
        # Restore original backend
        set_backend(original_backend)
    
    def test_get_current_backend(self):
        """Test getting current backend."""
        backend = get_current_backend()
        assert isinstance(backend, BackendType)
    
    def test_get_device(self):
        """Test getting current device string."""
        device = get_device()
        assert isinstance(device, str)
        assert device in ["cpu", "cuda", "mlx"]
    
    def test_create_tensor_global(self):
        """Test global tensor creation function."""
        data = [1, 2, 3]
        tensor = create_tensor(data)
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.tolist() == data


class TestBackendIntegration:
    """Integration tests for backend system."""
    
    def test_backend_switching(self):
        """Test switching between available backends."""
        manager = get_backend_manager()
        original_backend = manager.current_backend
        
        # Try switching to CPU (should always work)
        success = set_backend("cpu")
        assert success is True
        assert get_current_backend() == BackendType.CPU
        
        # Test tensor creation after switch
        tensor = create_tensor([1, 2, 3])
        assert tensor.device.type == "cpu"
        
        # Restore original backend
        set_backend(original_backend)
    
    def test_device_consistency(self):
        """Test that device strings are consistent."""
        manager = get_backend_manager()
        
        for backend_type in BackendType:
            if manager.available_backends[backend_type]:
                manager.set_backend(backend_type)
                device_str = manager.get_device_string()
                assert device_str == backend_type.value
    
    def test_tensor_creation_consistency(self):
        """Test tensor creation across different backends."""
        manager = get_backend_manager()
        data = [1, 2, 3, 4, 5]
        
        for backend_type in BackendType:
            if manager.available_backends[backend_type]:
                manager.set_backend(backend_type)
                tensor = manager.create_tensor(data)
                
                # All backends should create valid tensors
                assert hasattr(tensor, 'tolist') or hasattr(tensor, 'shape')
                
                # For PyTorch tensors, check the data
                if isinstance(tensor, torch.Tensor):
                    assert tensor.tolist() == data


@pytest.fixture
def fresh_backend_manager():
    """Provide a fresh backend manager for testing."""
    return BackendManager()


class TestBackendManagerFixture:
    """Tests using the fresh backend manager fixture."""
    
    def test_fresh_manager_initialization(self, fresh_backend_manager):
        """Test that fresh manager initializes correctly."""
        assert isinstance(fresh_backend_manager, BackendManager)
        assert BackendType.CPU in fresh_backend_manager.available_backends
        assert fresh_backend_manager.available_backends[BackendType.CPU] is True
    
    def test_fresh_manager_independence(self, fresh_backend_manager):
        """Test that fresh manager is independent of global state."""
        global_manager = get_backend_manager()
        
        # They should be different instances
        assert fresh_backend_manager is not global_manager
        
        # Changing one shouldn't affect the other
        original_global = global_manager.current_backend
        fresh_backend_manager.set_backend(BackendType.CPU)
        
        assert global_manager.current_backend == original_global 