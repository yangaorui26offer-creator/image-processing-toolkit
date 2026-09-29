# Image Processing Toolkit

An image processing project built with PyTorch and Flask. It provides a web interface for image classification, image denoising, and similar-image retrieval.

## Features

- **Image classification:** Recognizes five product categories: tops, shoes, bags, bottoms, and watches.
- **Image denoising:** Displays a noisy image alongside the model's denoised output.
- **Similar-image retrieval:** Finds similar images using encoder embeddings and cosine distance.

## Project structure

- `web/`: Flask service, page template, and interface assets.
- `image_classification/`: Classification model, training, and inference code.
- `image_denoising/`: Denoising model, training, and inference code.
- `image_similarity/`: Similarity model, pretrained weights, and image embeddings.
- `common/`: Shared utilities, image dataset, and labels.
- `test/`: Experiment scripts and notebooks.

The repository includes the current model weights, image dataset, and embedding files, so the initial download is relatively large.

## Run locally

Create and activate a Python virtual environment in the project root, then install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install flask torch torchvision numpy pillow scikit-learn matplotlib tqdm pandas ipython
```

The application uses relative paths. Start it from the `web` directory and add the project root to Python's module search path:

```bash
cd web
PYTHONPATH=.. python web_app.py
```

Then open http://127.0.0.1:9000 in a browser. Adjust dependency versions for your environment; the commands above are based on the existing code and have not been verified in a clean environment.

## GitHub repository page

The GitHub repository page is for viewing and managing the code. The interactive image-processing interface must be run locally or deployed to a separate server; publishing the repository does not automatically deploy the Flask application.
