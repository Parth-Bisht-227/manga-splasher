# manga-splasher🎨

An AI-powered web app that automatically colorizes black-and-white manga panels in a single click using deep learning. Just upload your manga image and get a vibrant color version in seconds!

## Features
- One-click manga panel colorization using Pix2Pix (U-Net generator + PatchGAN discriminator)
- Handles B&W line art with anime-inspired color palettes
- Clean Streamlit web interface—easy to run, demo, or deploy
- Model weights download securely from Google Drive on first use
- Training notebook included for reproducibility

## Architecture
- Generator: U-Net (encoder-decoder with skip connections)
- Discriminator: PatchGAN for fine-grained image discrimination
- Framework: TensorFlow/Keras
- Loss: Combined adversarial loss + L1 reconstruction loss

## Demo
(Streamlit Cloud app link here)

## How to use
1. Clone the repo:
   git clone https://github.com/Parth-Bisht-227/manga-splasher.git
   cd manga-splasher

2. Install requirements:
   pip install -r requirements.txt

3. Run locally:
   streamlit run app.py
   (Model weights are automatically downloaded for inference.)

## Project Structure
- app.py            # The Streamlit demo web app
- requirements.txt  # Python dependencies
- .streamlit/config.toml  # Custom theme (optional)
- checkpoints/      # Model weights folder (do not track, weights downloaded at runtime)
- notebooks/        # Training notebook (.ipynb), if added

## Training
See notebooks/manga_colorization_training.ipynb for the complete training pipeline, including:
- Data preprocessing and augmentation
- Pix2Pix model architecture (U-Net generator + PatchGAN discriminator)
- Training loop with adversarial and reconstruction losses
- Checkpoint management and inference code

## Technologies Used
- TensorFlow/Keras (deep learning framework)
- OpenCV (image processing)
- NumPy, Matplotlib (data handling and visualization)
- Streamlit (web interface)
- gdown (Google Drive weight downloads)

## License
MIT

## Credits
Created by Parth Bisht (@Parth-Bisht-227)
Model: Pix2Pix with U-Net generator and PatchGAN discriminator, adapted for manga colorization.
Built with TensorFlow, OpenCV, and Streamlit.


Feel free to fork, star, and use for creative manga or colorization projects!
🎨✨💖
