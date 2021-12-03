# Image group - Phase1 - Anomaly Detection

### To train and evaluate AUC on Constrained autoencoders

Setup:

    cd ConstrainedAEs
    bash download_dataset.sh
    pip install -r requirements.txt

Then to train an ordinary AE:

    python constrainedae.py 0

To train an ordinary VAE:

    python constrainedvae.py 0

To train an constrained AE:

    python constrainedae.py 0.1

To train an constrained VAE:

    python constrainedvae.py 0.1

The values passed in correspond to the ratio between the x loss and z loss, where a value of 0 is all x loss, aka image reconstruction, and 1 is all z loss, aka consistency of latent projection for image and image reconstruction.

