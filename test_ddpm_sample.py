import torch
import sys
import os

# Add the repo to Python path to import A2SB modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'nvidia-a2sb-original-repo'))

from A2SB_lightning_module_api import STFTBridgeModel

# Create a mock DDPM and VF model to make instantiation easy
class MockDDPM:
    def __init__(self):
        pass
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
    def forward(self, x_t, t_emb):
        return torch.ones_like(x_t)

def mock_compute_gaussian_product_coef(sigma_fwd, sigma_rev):
    return 1.0, 1.0, 1.0

# Mock it
import A2SB_lightning_module_api
A2SB_lightning_module_api.compute_gaussian_product_coef = mock_compute_gaussian_product_coef

class MockSTFTBridgeModel(STFTBridgeModel):
    def __init__(self):
        super(torch.nn.Module, self).__init__()
        self.ddpm = MockDDPM()
        self.vf_model = MockVFModel()
        self.use_ot_ode = False
    def t_to_emb(self, t):
        return torch.ones(t.shape[0], 1)

def test_ddpm_sample():
    model = MockSTFTBridgeModel()
    x_1 = torch.zeros(2, 3, 32, 32)
    t_steps = torch.linspace(1, 0, 11).unsqueeze(0).repeat(2, 1) # 10 steps

    # Test ddpm_sample
    all_pred_x0s = model.ddpm_sample(x_1, t_steps)
    assert len(all_pred_x0s) == 10

    # Test ddpm_sample_i2sb_way
    all_pred_x0s_2 = model.ddpm_sample_i2sb_way(x_1, t_steps)
    assert len(all_pred_x0s_2) == 10

    # Test ddpm_sample_i2sb_change_order
    all_pred_x0s_3 = model.ddpm_sample_i2sb_change_order(x_1, t_steps)
    assert len(all_pred_x0s_3) == 10

    print("Tests passed!")

if __name__ == '__main__':
    test_ddpm_sample()
