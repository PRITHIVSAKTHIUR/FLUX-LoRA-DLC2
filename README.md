
# FLUX LoRA DLC2 🔥

Experience the power of the **FLUX.1-dev** diffusion model combined with a massive collection of **100+ community-created LoRAs**! This Gradio application provides an easy-to-use interface to explore diverse artistic styles directly on top of the FLUX base model.

This "DLC Pack 2" builds upon the concept of easily accessible style enhancements (LoRAs) for the cutting-edge FLUX model. Generate stunning images, experiment with styles, and even load your own custom FLUX-compatible LoRAs from the Hugging Face Hub.

## Features

*   **FLUX.1-dev Base Model:** Utilizes the powerful `black-forest-labs/FLUX.1-dev` model.
*   **100+ Curated LoRAs:** A large, diverse gallery of pre-selected LoRAs with visual previews and trigger words (if applicable).
*   **Custom LoRA Support:** Load any FLUX.1-dev or FLUX.1-schnell compatible LoRA directly from a Hugging Face Hub repository URL or ID.
*   **Text-to-Image Generation:** Create images from text prompts combined with a selected LoRA style.
*   **Image-to-Image Generation:** Modify an existing image using a prompt and a LoRA style.
*   **Real-time Preview:** Leverages the tiny `TAEF1` VAE for quick previews during the generation steps (T2I only).
*   **High-Quality Final Output:** Uses the full `FLUX.1-dev` VAE for decoding the final, high-resolution image.
*   **Adjustable Parameters:** Control CFG scale, steps, image dimensions, seed, LoRA scale, and image strength (for I2I).
*   **User-Friendly Interface:** Simple Gradio UI for easy interaction.

## How to Use (Gradio Interface)

1.  **Select a LoRA:**
    *   Browse the **"100+ LoRA DLC's"** gallery and click on a style you like.
    *   The selected LoRA's Hugging Face repository link will appear above the gallery.
    *   The prompt placeholder will update, often suggesting the LoRA's title or trigger word.
    *   **OR** Enter a Hugging Face Hub repository ID (e.g., `username/repo-name`) or URL (e.g., `https://huggingface.co/username/repo-name`) into the **"Enter Custom LoRA"** textbox. The app will attempt to load and validate it.
2.  **Write Your Prompt:**
    *   Enter your desired image description in the "Prompt" box.
    *   **Important:** If the selected LoRA requires a specific **trigger word** (mentioned below the gallery selection or in the custom LoRA info card), make sure to include it in your prompt (usually at the beginning or end).
3.  **(Optional) Image-to-Image:**
    *   Expand the "Advanced Settings" section.
    *   Upload an **"Input image"**.
    *   Adjust the **"Denoise Strength"** slider. Lower values keep more of the original image, higher values allow more changes based on the prompt.
4.  **(Optional) Advanced Settings:**
    *   Adjust **CFG Scale**, **Steps**, **Width**, **Height**, **Seed**, and **LoRA Scale** as needed.
    *   Check/uncheck **"Randomize seed"**.
5.  **Generate:** Click the **"Generate"** button.
6.  **View Output:**
    *   For Text-to-Image, a progress bar and preview images will appear during generation.
    *   The final, high-quality image will be displayed in the "Generated Image" panel once complete.

## Custom LoRAs

You can load FLUX LoRAs that are not in the pre-defined list:

1.  Find a LoRA compatible with `black-forest-labs/FLUX.1-dev` or `black-forest-labs/FLUX.1-schnell` on the Hugging Face Hub.
2.  Paste the repository ID (e.g., `prithivMLmods/Canopus-LoRA-Flux-Anime`) or the full URL into the "Enter Custom LoRA" textbox.
3.  The application will attempt to:
    *   Verify the `base_model` in the LoRA's `README.md` or `model_index.json`.
    *   Find the `.safetensors` file within the repository.
    *   Fetch a preview image (if available).
    *   Extract trigger words (if defined in the metadata).
4.  If successful, an info card will appear, and the custom LoRA will be selected. Remember to include any necessary trigger words in your prompt.
5.  Use the "Remove custom LoRA" button to unload it and revert to the gallery selection.

*Note: Only LoRAs specifically trained for FLUX.1-dev or FLUX.1-schnell are expected to work correctly.*

## Technical Details

*   **Base Model:** `black-forest-labs/FLUX.1-dev`
*   **Preview VAE:** `madebyollin/taef1`
*   **Final VAE:** From `black-forest-labs/FLUX.1-dev`
*   **Core Library:** `diffusers`
*   **UI Framework:** `gradio`
*   **Backend:** `torch`

## Running Locally

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/PRITHIVSAKTHIUR/FLUX-LoRA-DLC2.git
    cd FLUX-LoRA-DLC2
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Ensure you have a compatible version of PyTorch installed, preferably with CUDA support if you have an NVIDIA GPU.* (See [pytorch.org](https://pytorch.org/))
4.  **(Optional) Hugging Face Login:** For potentially faster downloads or access to private models (if ever needed):
    ```bash
    huggingface-cli login
    # Or set environment variable: export HF_TOKEN=your_token
    ```
5.  **Run the application:**
    ```bash
    python app.py
    ```
6.  Open your web browser and navigate to the local URL provided (usually `http://127.0.0.1:7860`).

## Dependencies

*   `torch`
*   `diffusers`
*   `gradio`
*   `transformers` (likely a sub-dependency of diffusers)
*   `accelerate`
*   `numpy`
*   `Pillow`
*   `huggingface-hub`
*   `requests` (for fetching images, often a sub-dependency)

See `requirements.txt` for specific versions.

```bash
# requirements.txt
torch --index-url https://download.pytorch.org/whl/cu121 # Or adjust for your CUDA/CPU version
diffusers>=0.29.0 # Check for latest compatible version
gradio
transformers
accelerate
numpy
Pillow
huggingface-hub
requests
# Optional: Add specific versions if needed based on compatibility
# Example: diffusers==0.29.0
```

*(Note: Create the `requirements.txt` file with the content above, adjusting the PyTorch line based on your system (CUDA version or CPU) if necessary.)*

## Acknowledgements

*   **Black Forest Labs:** For creating the amazing FLUX models.
*   **Hugging Face:** For the `diffusers`, `transformers`, `huggingface_hub` libraries and the Spaces platform.
*   **Ollin:** For the `TAEF1` tiny autoencoder used for previews.
*   **Gradio Team:** For the easy-to-use Gradio framework.
*   **All LoRA Creators:** A huge thank you to the talented individuals and teams who created the LoRAs featured in this application! Your work makes exploring diverse styles incredibly accessible.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details. (You'll need to add an Apache 2.0 LICENSE file to your repo).
```

**Next Steps:**

1.  **Save:** Save this content as `README.md` in the root of your `FLUX-LoRA-DLC2` repository.
2.  **Create `requirements.txt`:** Create a file named `requirements.txt` in the root of your repository with the dependency list provided in the README section. Adjust the `torch` line if you need a specific CUDA version or the CPU version.
3.  **Add License:** Add a file named `LICENSE` containing the standard [Apache 2.0 License text](https://www.apache.org/licenses/LICENSE-2.0.txt).
4.  **Update HF Space Badge:** If you deploy this on Hugging Face Spaces, replace `YOUR_HF_USERNAME/YOUR_SPACE_NAME` in the badge URL with your actual Space details.
5.  **Commit and Push:** Add these files to your git repository, commit, and push them to GitHub.
