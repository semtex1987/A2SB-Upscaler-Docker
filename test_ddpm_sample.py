import torch
import sys
import os
import unittest.mock

# Add the repo to Python path to import A2SB modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'nvidia-a2sb-original-repo'))

sys.modules['lightning'] = unittest.mock.MagicMock()
sys.modules['lightning.pytorch'] = unittest.mock.MagicMock()
class MockLightningModule(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
sys.modules['lightning.pytorch'].LightningModule = MockLightningModule

sys.modules['torchaudio'] = unittest.mock.MagicMock()
sys.modules['torchaudio.transforms'] = unittest.mock.MagicMock()
sys.modules['librosa'] = unittest.mock.MagicMock()
sys.modules['numpy'] = unittest.mock.MagicMock()
sys.modules['tqdm'] = unittest.mock.MagicMock()
sys.modules['einops'] = unittest.mock.MagicMock()
sys.modules['pandas'] = unittest.mock.MagicMock()
sys.modules['scipy'] = unittest.mock.MagicMock()
sys.modules['rotary_embedding_torch'] = unittest.mock.MagicMock()

class MockMatplotlib:
    pass

sys.modules['matplotlib'] = unittest.mock.MagicMock()
sys.modules['matplotlib.pyplot'] = unittest.mock.MagicMock()
sys.modules['matplotlib.pylab'] = unittest.mock.MagicMock()
sys.modules['matplotlib.backends'] = unittest.mock.MagicMock()
sys.modules['matplotlib.backends.backend_agg'] = unittest.mock.MagicMock()
sys.modules['moviepy'] = unittest.mock.MagicMock()
sys.modules['moviepy.video'] = unittest.mock.MagicMock()
sys.modules['moviepy.video.io'] = unittest.mock.MagicMock()
sys.modules['moviepy.video.io.bindings'] = unittest.mock.MagicMock()
sys.modules['jsonargparse'] = unittest.mock.MagicMock()
sys.modules['pesq'] = unittest.mock.MagicMock()
sys.modules['ssr_eval'] = unittest.mock.MagicMock()
sys.modules['scipy.io'] = unittest.mock.MagicMock()
sys.modules['scipy.io.wavfile'] = unittest.mock.MagicMock()

from A2SB_lightning_module_api import STFTBridgeModel

# Create a mock DDPM and VF model to make instantiation easy
class MockDDPM(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def get_pred_x0(self, t, x_t, vf_output):
        return x_t + vf_output
    def p_posterior(self, t_prev, t, x_t, pred_x0, ot_ode=False):
        return x_t - pred_x0 * 0.1
    def get_std_t(self, t):
        return torch.ones_like(t) * 0.1
    def get_std_fwd(self, t):
        return torch.ones_like(t) * 0.1
    def get_std_rev(self, t):
        return torch.ones_like(t) * 0.1

class MockVFModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x_t, t_emb):
        return torch.ones_like(x_t)

def mock_compute_gaussian_product_coef(sigma_fwd, sigma_rev):
    return torch.tensor(1.0, dtype=sigma_fwd.dtype, device=sigma_fwd.device), \
           torch.tensor(1.0, dtype=sigma_fwd.dtype, device=sigma_fwd.device), \
           torch.tensor(1.0, dtype=sigma_fwd.dtype, device=sigma_fwd.device)

# Mock it
import A2SB_lightning_module_api
A2SB_lightning_module_api.compute_gaussian_product_coef = mock_compute_gaussian_product_coef

import A2SB_lightning_module
A2SB_lightning_module.compute_gaussian_product_coef = mock_compute_gaussian_product_coef

class MockSTFTBridgeModel(STFTBridgeModel):
    def __init__(self):
        # We call the proper initializer explicitly as requested
        # First we patch what we need to safely initialize
        with unittest.mock.patch.object(STFTBridgeModel, 'configure_optimizers', return_value=None):
            STFTBridgeModel.__init__(self, vf_model=unittest.mock.MagicMock())
        self.ddpm = MockDDPM()
        self.vf_model = MockVFModel()
        self.use_ot_ode = False
    def t_to_emb(self, t):
        return torch.ones(t.shape[0], 1)

def verify_output(outputs, expected_len, shape):
    assert len(outputs) == expected_len
    for idx, x in enumerate(outputs):
        assert x is not None, f"Found None at index {idx}"
        assert x.shape == shape, f"Expected shape {shape}, got {x.shape} at index {idx}"
        assert torch.isfinite(x).all(), f"Found non-finite values at index {idx}"

def test_ddpm_sample():
    model = MockSTFTBridgeModel()
    x_1 = torch.zeros(2, 3, 32, 32)
    t_steps = torch.linspace(1, 0, 11).unsqueeze(0).repeat(2, 1) # 10 steps

    # Test ddpm_sample
    all_pred_x0s = model.ddpm_sample(x_1, t_steps)
    verify_output(all_pred_x0s, 10, x_1.shape)

    # Test ddpm_sample_i2sb_way
    all_pred_x0s_2 = model.ddpm_sample_i2sb_way(x_1, t_steps)
    verify_output(all_pred_x0s_2, 10, x_1.shape)

    # Test ddpm_sample_i2sb_change_order
    all_pred_x0s_3 = model.ddpm_sample_i2sb_change_order(x_1, t_steps)
    verify_output(all_pred_x0s_3, 10, x_1.shape)

    print("Tests passed!")

if __name__ == '__main__':
    test_ddpm_sample()
