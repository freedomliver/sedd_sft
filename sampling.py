import abc
import torch
import torch.nn.functional as F
from catsample import sample_categorical

from model import utils as mutils

_PREDICTORS = {}


def _choose_tokens(probs, sampling_mode):
    if sampling_mode == "sample":
        return sample_categorical(probs)
    if sampling_mode == "argmax":
        return probs.argmax(dim=-1)
    raise ValueError(f"Unknown sampling_mode: {sampling_mode}")


def register_predictor(cls=None, *, name=None):
    """A decorator for registering predictor classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _PREDICTORS:
            raise ValueError(
                f'Already registered model with name: {local_name}')
        _PREDICTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)

    
def get_predictor(name):
    return _PREDICTORS[name]



class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, graph, noise):
        super().__init__()
        self.graph = graph
        self.noise = noise

    @abc.abstractmethod
    def update_fn(self, score_fn, x, t, step_size):
        """One update of the predictor.

        Args:
            score_fn: score function
            x: A PyTorch tensor representing the current state
            t: A Pytorch tensor representing the current time step.

        Returns:
            x: A PyTorch tensor of the next state.
        """
        pass


@register_predictor(name="euler")
class EulerPredictor(Predictor):
    def update_fn(self, score_fn, x, t, step_size):
        sigma, dsigma = self.noise(t)
        score = score_fn(x, sigma)

        rev_rate = step_size * dsigma[..., None] * self.graph.reverse_rate(x, score)
        x = self.graph.sample_rate(x, rev_rate)
        return x

@register_predictor(name="none")
class NonePredictor(Predictor):
    def update_fn(self, score_fn, x, t, step_size):
        return x


@register_predictor(name="analytic")
class AnalyticPredictor(Predictor):
    def update_fn(self, score_fn, x, t, step_size, sampling_mode="sample"):
        curr_sigma = self.noise(t)[0]
        next_sigma = self.noise(t - step_size)[0]
        dsigma = curr_sigma - next_sigma

        score = score_fn(x, curr_sigma)

        stag_score = self.graph.staggered_score(score, dsigma)
        probs = stag_score * self.graph.transp_transition(x, dsigma)
        return _choose_tokens(probs, sampling_mode)

    
class Denoiser:
    def __init__(self, graph, noise):
        self.graph = graph
        self.noise = noise

    def update_fn(self, score_fn, x, t, sampling_mode="sample"):
        sigma = self.noise(t)[0]

        score = score_fn(x, sigma)
        stag_score = self.graph.staggered_score(score, sigma)
        probs = stag_score * self.graph.transp_transition(x, sigma)
        # truncate probabilities
        if self.graph.absorb:
            probs = probs[..., :-1]
        
        return _choose_tokens(probs, sampling_mode)
                       

def get_sampling_fn(config, graph, noise, batch_dims, eps, device):
    
    sampling_fn = get_pc_sampler(graph=graph,
                                 noise=noise,
                                 batch_dims=batch_dims,
                                 predictor=config.sampling.predictor,
                                 steps=config.sampling.steps,
                                 denoise=config.sampling.noise_removal,
                                 eps=eps,
                                 device=device)
    
    return sampling_fn
    

def get_pc_sampler(
    graph,
    noise,
    batch_dims,
    predictor,
    steps,
    denoise=True,
    eps=1e-5,
    device=torch.device('cpu'),
    proj_fun=lambda x: x,
    init_x=None,
    sampling_mode="sample",
):
    predictor = get_predictor(predictor)(graph, noise)
    projector = proj_fun
    denoiser = Denoiser(graph, noise)

    @torch.no_grad()
    def pc_sampler(model):
        sampling_score_fn = mutils.get_score_fn(model, train=False, sampling=True)
        if init_x is None:
            x = graph.sample_limit(*batch_dims).to(device)
        else:
            x = init_x.to(device)
        x = projector(x)
        timesteps = torch.linspace(1, eps, steps + 1, device=device)
        dt = (1 - eps) / steps

        for i in range(steps):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
            x = projector(x)
            if isinstance(predictor, AnalyticPredictor):
                x = predictor.update_fn(sampling_score_fn, x, t, dt, sampling_mode)
            else:
                x = predictor.update_fn(sampling_score_fn, x, t, dt)
            x = projector(x)
            

        if denoise:
            # denoising step
            x = projector(x)
            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=device)
            x = denoiser.update_fn(sampling_score_fn, x, t, sampling_mode)
            x = projector(x)
            
        return x
    
    return pc_sampler




def get_prompt_clamped_sampler(
    graph,
    noise,
    prompt_ids,
    max_answer_len=512,
    max_length=1024,
    steps=128,
    predictor="analytic",
    denoise=True,
    eps=1e-4,
    device="cuda",
    sampling_mode="sample",
):
    @torch.no_grad()
    def sampler(model):
        prompt = prompt_ids.to(device)
        B, prompt_len = prompt.shape
        answer_len = min(max_answer_len, max_length - prompt_len)
        if answer_len <= 0:
            raise ValueError("prompt is too long for the requested max_length")

        x_answer = graph.sample_limit(B, answer_len).to(device)
        x = torch.cat([prompt, x_answer], dim=1)

        def clamp_prompt(y):
            y = y.clone()
            y[:, :prompt_len] = prompt
            return y

        sampling_fn = get_pc_sampler(
            graph=graph,
            noise=noise,
            batch_dims=(B, prompt_len + answer_len),
            predictor=predictor,
            steps=steps,
            denoise=denoise,
            eps=eps,
            device=device,
            proj_fun=clamp_prompt,
            init_x=x,
            sampling_mode=sampling_mode,
        )
        return sampling_fn(model)

    return sampler


def get_conditional_sampler(graph, noise, question_ids, steps=128, eps=1e-4, device="cuda"):
    sampler = get_prompt_clamped_sampler(
        graph=graph,
        noise=noise,
        prompt_ids=question_ids,
        max_answer_len=1024 - question_ids.shape[1],
        max_length=1024,
        steps=steps,
        predictor="analytic",
        denoise=True,
        eps=eps,
        device=device,
    )
    return sampler
