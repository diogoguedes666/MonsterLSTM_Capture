import pytest
import torch
from dist_model_recnet import AutoTuner, AutoTunerConfig

def test_autotuner_initialization():
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters())
    config = AutoTunerConfig(initial_lr=0.001, initial_grad_clip=5.0)
    
    tuner = AutoTuner(optimizer, config)
    assert tuner.current_lr == config.initial_lr
    assert tuner.current_grad_clip == config.initial_grad_clip

def test_invalid_config():
    with pytest.raises(ValueError):
        config = AutoTunerConfig(initial_lr=-1, initial_grad_clip=5.0)

def test_learning_rate_adjustment():
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters())
    config = AutoTunerConfig(initial_lr=0.001, initial_grad_clip=5.0)
    tuner = AutoTuner(optimizer, config)
    
    # Test LR reduction
    old_lr = tuner.current_lr
    tuner.reduce_learning_rate()
    assert tuner.current_lr < old_lr 