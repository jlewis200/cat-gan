#!/usr/bin/python3

"""
A program to perform GAN experiments.
"""


import math
from argparse import ArgumentParser
from time import time
from contextlib import nullcontext

# pylint: disable=no-member
import torch

from torch.utils.data import DataLoader
from torch.nn.functional import interpolate
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms.functional import to_tensor, to_pil_image
from PIL import Image
from configs import get_model
from model import GanOptimizer

GRAD_CLIP = 2 ** 32
EMA_SCALER = 0.99
SAVE_INTERVAL = 3600


def get_args(args=None):
    """
    Parse command line options.
    """

    parser = ArgumentParser(description="Perform GAN experiments")
    parser.add_argument("-t", "--train", action="store_true", help="train the model")
    parser.add_argument(
        "-l", "--learning-rate", type=float, default=0.00005, help="learning rate"
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=50, help="number of training epochs"
    )
    parser.add_argument("-n", "--batch-size", type=int, default=8, help="batch size")
    parser.add_argument(
        "-d", "--device", type=str, default="cpu", help="torch device string"
    )
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--config", default="cat-faces", type=str)
    parser.add_argument("--sample", type=int, help="number of samples")
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="temperature of samples"
    )
    return parser.parse_args(args=args)


def main(args):
    """
    Main function to initiate training/experiments.
    """

    model = get_model(args.config)

    # ensure a valid config was specified
    if model is None:
        print("config not found")
        return

    # try sending a tensor to specified device to check validity/availability
    try:
        torch.zeros(1, device=args.device)

    except RuntimeError:
        print(f"inavlid/unavailable device specified:  {args.device}")
        return

    model = model.to(args.device)
    optimizer = GanOptimizer(model, args.learning_rate)

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        optimizer.gen.load_state_dict(checkpoint["optimizer_gen_state_dict"])
        optimizer.dis.load_state_dict(checkpoint["optimizer_dis_state_dict"])
        optimizer.set_lr(args.learning_rate)

    if args.train:
        model = train(
            model=model,
            optimizer=optimizer,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )

    if args.sample is not None:
        sample(model, args.sample, temp=args.temperature)


def get_montage(imgs):
    """
    Build a montage from a list of PIL images.
    """

    montage = Image.new("RGB", (imgs[0].shape[-1] * len(imgs), imgs[0].shape[-2]))

    for idx, img in enumerate(imgs):
        img = img.squeeze(0)
        img = to_pil_image(img)
        montage.paste(img, (idx * imgs[0].shape[-1], 0))

    return montage


def sample(model, n_samples, temp=1.0):
    """
    Sample from the model.
    """

    model.eval()
    with torch.no_grad():
        samples = [(model.sample(1, temp=temp)).clamp(0, 1) for _ in range(n_samples)]

    samples = [interpolate(sample, 256) for sample in samples]
    get_montage(samples).save("samples.jpg")


def train(model, optimizer, epochs=50, batch_size=32):
    """
    Train the model using supplied hyperparameters.
    train_enc_dec controls whether the encoder and decoder should be trained, or only the dmll net.
    """

    model.dataset = model.get_dataset()
    dataloader = DataLoader(
        model.dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        prefetch_factor=2,
        drop_last=True,
    )

    # initialize exponential moving averages
    loss_dis_ema, loss_gen_ema, grad_norm_dis_ema, grad_norm_gen_ema = (torch.inf,) * 4

    for _ in range(epochs):
        model.epoch += 1

        print(
            f"{'':>9} {'sample/':>9} {'grad norm':>9} {'grad norm':>9} {'loss':>9} {'loss':>9}"
        )
        print(f"{'epoch':>9} {'sec':>9} {'dis':>9} {'gen':>9} {'dis':>9} {'gen':>9}")

        samples = 0
        epoch_start = time()

        for _ in range(len(dataloader)):
            loss_dis, grad_norm_dis = dis_step(model, dataloader, optimizer)
            loss_gen, grad_norm_gen = gen_step(model, batch_size, optimizer)

            if all(
                not math.isnan(item)
                for item in (loss_dis, loss_gen, grad_norm_dis, grad_norm_gen)
            ):
                samples += batch_size
                loss_dis_ema = ema_update(loss_dis_ema, loss_dis)
                loss_gen_ema = ema_update(loss_gen_ema, loss_gen)
                grad_norm_dis_ema = ema_update(grad_norm_dis_ema, grad_norm_dis)
                grad_norm_gen_ema = ema_update(grad_norm_gen_ema, grad_norm_gen)

            samples_sec = samples / (time() - epoch_start)

            print(
                f"{model.epoch.item():9} "
                f"{samples_sec: 9.2e} "
                f"{grad_norm_dis: 9.2e} "
                f"{grad_norm_gen: 9.2e} "
                f"{loss_dis: 9.2e} "
                f"{loss_gen: 9.2e} "
            )

            print(
                f"{'':9} "
                f"{' EMAs':9} "
                f"{grad_norm_dis_ema: 9.2e} "
                f"{grad_norm_gen_ema: 9.2e} "
                f"{loss_dis_ema: 9.2e} "
                f"{loss_gen_ema: 9.2e} "
            )

            print()

        save_state(model, optimizer)

    return model


def dis_step(model, dataloader, optimizer, iterations=5):
    """
    Take one training step for a given batch of data.
    Use CUDA-specific FP16 optimizations if using a CUDA device.
    """

    loss_sum, grad_norm_sum = 0, 0

    #if model.gen_steps < 25 or model.gen_steps % 500 == 0:
    if model.gen_steps < 25:
        iterations = 100

    with torch.cuda.amp.autocast() if model.device.type == "cuda" else nullcontext():
        for idx, batch in enumerate(dataloader):
            if idx > iterations:
                break

            batch = batch[0].to(model.device)
            batch_score = model.forward(samples=batch).mean()
            sample_score = model.forward(n_samples=batch.shape[0]).mean()

            loss = batch_score - sample_score
            loss_sum += loss.item()
            optimizer.dis_backward(loss)

            grad_norm_sum += clip_grad_norm_(model.parameters(), GRAD_CLIP).item()

            optimizer.dis_step()
            optimizer.zero_grad()

            model.clip_params()

    return loss_sum / iterations, grad_norm_sum / iterations


def gen_step(model, n_samples, optimizer):
    """
    Take one training step for a given batch of data.
    """

    loss = model.forward(n_samples=n_samples).mean()
    loss.backward()

    grad_norm = clip_grad_norm_(model.parameters(), GRAD_CLIP)

    optimizer.gen_step()
    optimizer.zero_grad()
    model.gen_steps += 1

    return loss.item(), grad_norm.item()


def ema_update(ema, val):
    """
    Perform an ema update given a new value.
    """

    if ema == torch.inf:
        # ema is initialized with inf
        ema = val

    else:
        ema = (ema * EMA_SCALER) + ((1 - EMA_SCALER) * val)

    return ema


def save_state(model, optimizer):
    """
    Save the model/optimizer state dict.
    """

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_gen_state_dict": optimizer.gen.state_dict(),
            "optimizer_dis_state_dict": optimizer.dis.state_dict(),
        },
        f"checkpoints/model_{model.start_time.item()}_{model.epoch.item()}.pt",
    )


if __name__ == "__main__":
    main(get_args())
