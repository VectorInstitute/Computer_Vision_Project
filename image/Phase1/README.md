# Image group - Phase1 - Anomaly Detection

To train and evaluate AUC on Constrained autoencoders:

    cd ConstrainedAEs
    bash download_dataset.sh
    pip install -r requirements.txt

then to train an ordinary AE:

    python constrainedae.py 0

to train an ordinary VAE:

    python constrainedvae.py 0

to train an constrained AE:

    python constrainedae.py 0.1

to train an constrained VAE:

    python constrainedvae.py 0.1

The values passed in corrispond to the ratio between the x loss and z loss, where a value of 0 is all x loss, aka image reconstruction, and 1 is all z loss, aka consistency of latent projection for image and image reconstruction.

