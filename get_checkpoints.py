#!/usr/bin/python3

"""
Script for downloading the pretrained model weights from google drive.
"""

import gdown


def main():
    """
    Download the pretrained model weights.
    """

    try:
        # cifar10 pretrained
        print("attempting to download cat faces pretrained weights")
        url = "https://drive.google.com/file/d/1aSmkamN1SL_Nx8cmW9IKNOvAfhm4J1DZ/view?usp=sharing"
        destination = "checkpoints/cat_faces_pretrained.pt"
        gdown.download(url, destination, quiet=False, fuzzy=True)
        print()

    except FileNotFoundError:
        print("missing 'checkpoints' directory\n")

    except RuntimeError:
        print("unable to download cat faces pretrained weights\n")


if __name__ == "__main__":
    main()
