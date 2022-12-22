import argparse, os
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
from imwatermark import WatermarkEncoder

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

torch.set_grad_enabled(False)

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--dpm",
        action='store_true',
        help="use DPM (2) sampler",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across all samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=3,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=3,
        help="how many samples to produce for each given prompt. A.k.a batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=9.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file, separated by newlines",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v2-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="repeat each prompt in file this often",
    )
    opt = parser.parse_args()
    return opt


class sd_gen():
    def __init__(
        self, 
        config="/DATA/wenyan/stablediffusion/configs/stable-diffusion/v2-inference.yaml",
        ckpt="/DATA/wenyan/stablediffusion/v2-1_512-ema-pruned.ckpt",
        plms:bool=True,
        dpm:bool=False,
        H:int=512,
        W:int=512,
        C:int=4,
        f:int=8,
        batch_size=4,
        n_iter=1,
        steps=50,
        scale=9.0,
        ddim_eta=0.0,
        precision="autocast", # or "full"
        fixed_code=False,
        seed=42
    ):
        self.config = OmegaConf.load(f"{config}")
        self.plms = plms
        self.dpm = dpm
        self.ckpt = ckpt
        self.H = H
        self.W = W
        self.C = C
        self.f = f
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.steps = steps
        self.scale = scale
        self.ddim_eta = ddim_eta
        self.precision = precision
        self.fixed_code = fixed_code

        seed_everything(seed)


    def generate(self, data, outdir="outputs/txt2img-samples"):
        # init model
        model = load_model_from_config(self.config, f"{self.ckpt}")
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model = model.to(device)

        # init sampler
        if self.plms:
            sampler = PLMSSampler(model)
        elif self.dpm:
            sampler = DPMSolverSampler(model)
        else:
            sampler = DDIMSampler(model)

        os.makedirs(outdir, exist_ok=True)
        sample_path = outdir

        batch_size = self.batch_size
        
        batched_data = list(chunk(data, batch_size))
        # print("Data: ", data)

        sample_count = 0
        base_count = len(os.listdir(sample_path))


        start_code = None
        if self.fixed_code:
            start_code = torch.randn([self.batch_size, self.C, self.H // self.f, self.W // self.f], device=device)

        precision_scope = autocast if self.precision == "autocast" else nullcontext
        with torch.no_grad(), \
            precision_scope("cuda"), \
            model.ema_scope():
                all_samples = list()
                for n in trange(self.n_iter, desc="Sampling"):
                    for batch in tqdm(batched_data, desc="data"):
                        # print("batch: ", batch)
                        uc = None
                        batch_size = min(len(batch), batch_size)
                        if self.scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(batch, tuple):
                            prompts = [x["caption"] for x in batch]

                        c = model.get_learned_conditioning(prompts)

                        shape = [self.C, self.H // self.f, self.W // self.f]
                        samples, _ = sampler.sample(S=self.steps,
                                                    conditioning=c,
                                                    batch_size=batch_size,
                                                    shape=shape,
                                                    verbose=False,
                                                    unconditional_guidance_scale=self.scale,
                                                    unconditional_conditioning=uc,
                                                    eta=self.ddim_eta,
                                                    x_T=start_code)

                        x_samples = model.decode_first_stage(samples)
                        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                        for idx, x_sample in enumerate(x_samples):
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            img = Image.fromarray(x_sample.astype(np.uint8))
                            img.save(os.path.join(sample_path, batch[idx]["image"]))
                            base_count += 1
                            sample_count += 1

                        all_samples.append(x_samples)

        # free up memory after use
        # model=model.cpu()
        del model



def main():
    # data = [
    #     "a man in a purple shirt is sitting on a stool in front of a crowd", 
    #     "a hockey player in a blue and yellow uniform is standing in front of a goal",
    #     "a man with green hair and a green wig is yelling",
    #     "a clown in a red dress and blue wig is walking down the street"
    #     ]

    data = [
        {"image": "4761802461.jpg", "caption": "a picture of the olde english dance troupe exhibits to park goers or festival attendees formal dress fashion, dancing styles and steps that were prevalent in court and formal balls before the 20th", "source": "original", "loss_val": 210.50729370117188, "caption_id": 0, "sample_id": "4761802461_0"}, 
        {"image": "8117746605.jpg", "caption": "a picture of shane battier and davonik nastavowich communicate and try to crack a smile before the up coming game against their rivals miami heat in the 2011 nba championship game", "source": "original", "loss_val": 208.052490234375, "caption_id": 0, "sample_id": "8117746605_0"}, 
        {"image": "5969756753.jpg", "caption": "a picture of rock band def leppard, pictured counterclockwise include rick savage, vivian campbell, joe elliot, and phil collen, looking up to drummer rick allen on an elevated stage platform during a concert", "source": "original", "loss_val": 207.21560668945312, "caption_id": 0, "sample_id": "5969756753_0"}, 
        {"image": "7421074812.jpg", "caption": "a picture of 3 man on a boot one with dark sungalsses and gray hair and gray shirt sitting inside the building, one going has very little hair green and white checked shirt", "source": "original", "loss_val": 204.24578857421875, "caption_id": 0, "sample_id": "7421074812_0"}]
        
    sd = sd_gen(batch_size=8)
    sd.generate(data)

   
if __name__ == "__main__":
    main()
