# Partly from https://github.com/fastai/fastdiffusion/blob/master/nbs/tcapelle/Diffusion_models_with_fastai_conditional_cifart_EMA.ipynb
from fastai.vision.all import *
from fastai.vision.gan import *
from fastai.data import *
from copy import deepcopy
import torch

from unet import Unet
from config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConditionalDDPMCallback(Callback):
    def __init__(self, n_steps, beta_min, beta_max, cfg_scale=0):
        store_attr()
        self.tensor_type = TensorImage

    def before_fit(self):
        self.beta = torch.linspace(self.beta_min, self.beta_max, self.n_steps).to(
            self.dls.device)  # variance schedule, linearly increased with timestep
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.sigma = torch.sqrt(self.beta)

    def sample_timesteps(self, x, dtype=torch.long):
        return torch.randint(self.n_steps, (x.shape[0],), device=x.device, dtype=dtype)

    def generate_noise(self, x):
        return self.tensor_type(torch.randn_like(x))

    def noise_image(self, x, eps, t):
        alpha_bar_t = self.alpha_bar[t][:, None, None, None]
        return torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * eps  # noisify the image

    def before_batch_training(self):
        x0 = self.xb[0]  # original images and labels
        y0 = self.yb[0] if np.random.random() > 0.1 else None

        # y0 = None

        eps = self.generate_noise(x0)  # noise same shape as x0
        t = self.sample_timesteps(x0)  # select random timesteps
        xt = self.noise_image(x0, eps, t)  # add noise to the image
        # print(x0.shape, y0.shape, t.shape, xt.shape, eps.shape)

        self.learn.xb = (xt, t, y0)  # input to our model is noisy image, timestep and label
        self.learn.yb = (eps,)  # ground truth is the noise

    def sampling_algo(self, xt, t, label=None):
        t_batch = torch.full((xt.shape[0],), t, device=xt.device, dtype=torch.long)
        z = self.generate_noise(xt) if t > 0 else torch.zeros_like(xt)
        alpha_t = self.alpha[t]  # get noise level at current timestep
        alpha_bar_t = self.alpha_bar[t]
        sigma_t = self.sigma[t]
        alpha_bar_t_1 = self.alpha_bar[t - 1] if t > 0 else torch.tensor(1, device=xt.device)
        beta_bar_t = 1 - alpha_bar_t
        beta_bar_t_1 = 1 - alpha_bar_t_1
        predicted_noise = self.model(xt, t_batch, label=label)
        if self.cfg_scale > 0:
            uncond_predicted_noise = self.model(xt, t_batch, label=None)
            predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, self.cfg_scale)
        x0hat = (xt - torch.sqrt(beta_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
        x0hat = torch.clamp(x0hat, -1, 1)
        xt = x0hat * torch.sqrt(alpha_bar_t_1) * (1 - alpha_t) / beta_bar_t + xt * torch.sqrt(
            alpha_t) * beta_bar_t_1 / beta_bar_t + sigma_t * z

        return xt

    # def sampling_algo_old(self, xt, t, label=None):
    #     t_batch = torch.full((xt.shape[0],), t, device=xt.device, dtype=torch.long)
    #     z = self.generate_noise(xt) if t > 0 else torch.zeros_like(xt)
    #     alpha_t = self.alpha[t] # get noise level at current timestep
    #     alpha_bar_t = self.alpha_bar[t]
    #     sigma_t = self.sigma[t]
    #     xt = 1/torch.sqrt(alpha_t) * (xt - (1-alpha_t)/torch.sqrt(1-alpha_bar_t) * self.model(xt, t_batch, label=label)) + sigma_t*z
    #          1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
    #     # predict x_(t-1) in accordance to Algorithm 2 in paper
    #     return xt

    def get_sample(self, input_dim, label, embedding=None):
        if embedding is None:
            embedding = self.tensor_type(torch.randn(input_dim))
        embedding = embedding.to(device)
        for t in progress_bar(reversed(range(self.n_steps)), total=self.n_steps, leave=False):
            images = self.sampling_algo(embedding, t, label)

        return images, embedding

    def before_batch_sampling(self):
        xt = self.sample()
        self.learn.pred = (xt,)
        raise CancelBatchException

    def after_validate(self):
        if (self.epoch + 1) % 4 == 0:
            with torch.no_grad():
                xt = self.sample()

    def before_batch(self):
        if not hasattr(self, 'gather_preds'):
            self.before_batch_training()
        else:
            self.before_batch_sampling()

    def get_sample(self, input_dim, label):
        xt = self.tensor_type(torch.randn(input_dim))
        xt = xt.to(device)
        for t in progress_bar(reversed(range(self.n_steps)), total=self.n_steps, leave=False):
            xt = self.sampling_algo(xt, t, label)
        return xt

    def save(self, filepath):
        state = {
            'n_steps': self.n_steps,
            'beta_min': self.beta_min,
            "beta": self.beta,
            'beta_max': self.beta_max,
            'cfg_scale': self.cfg_scale,
            'alpha': self.alpha,
            "alpha_bar": self.alpha_bar,
            "sigma": self.sigma,
        }
        torch.save(state, filepath)

    def load(self, filepath):
        state = torch.load(filepath)
        self.n_steps = state['n_steps']
        self.beta_min = state['beta_min']
        self.beta_max = state['beta_max']
        self.cfg_scale = state['cfg_scale']
        self.alpha = state['alpha']
        self.alpha_bar = state["alpha_bar"]
        self.sigma = state["sigma"]
        self.beta = state["beta"]


class EMA(Callback):
    "Exponential Moving average CB"

    def __init__(self, beta=0.995, pct_start=0.3):
        store_attr()

    def before_fit(self):
        self.ema_model = deepcopy(self.model).eval().requires_grad_(False)
        self.step_start_ema = int(self.pct_start * self.n_epoch)  # start EMA at 30% of epochs

    def update_model_average(self):
        for current_params, ma_params in zip(self.model.parameters(), self.ema_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self):
        if self.epoch < self.step_start_ema:
            self.reset_parameters()
            self.step += 1
            return
        self.update_model_average()
        self.step += 1

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def after_batch(self):
        if hasattr(self, 'pred'): return
        self.step_ema()

    def after_training(self):
        self.model = self.ema_model


@delegates(Unet)
class ConditionalUnet(Unet):
    def __init__(self, dim, num_classes=None, **kwargs):
        super().__init__(dim=dim, **kwargs)
        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, dim * 4)

    def forward(self, x, time, label=None):
        x = self.init_conv(x)
        t = self.time_mlp(time)
        if label is not None:
            t += self.label_emb(label)

        return super().forward_blocks(x, t)
