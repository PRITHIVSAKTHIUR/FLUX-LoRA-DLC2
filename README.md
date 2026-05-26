# **[FLUX-LoRA-DLC2](https://huggingface.co/spaces/prithivMLmods/FLUX-LoRA-DLC2)**

FLUX-LoRA-DLC2 is an experimental, advanced image generation and image-to-image manipulation ecosystem. Built on top of the state-of-the-art `black-forest-labs/FLUX.1-dev` foundation, this application incorporates a dynamic multi-LoRA switching engine loaded with a comprehensive collection of over 100 stylistic adapters (such as Sin City Movie, Claymation XC, and Vector Flux styles). The environment introduces an adaptive decoding layout utilizing `madebyollin/taef1` for streaming latent previews and a localized high-fidelity VAE for final image synthesis. Featuring custom quality expansion presets, automated token blending rules, and direct Hugging Face repository scanning for custom third-party adapter loading, FLUX-LoRA-DLC2 functions as a robust sandbox for pushing artistic boundaries in generative artificial intelligence.

https://github.com/user-attachments/assets/891a0ad2-caad-4376-852e-4b8aeee0ea5e

### **Key Features**

* **Dynamic LoRA Loader & Selector:** Seamlessly browse and activate over 100 pre-configured stylistic LoRAs via an interactive structural grid gallery, or fetch any custom validation model directly by pasting its Hugging Face repository path.
* **Dual Inference Strategies:** Fully supports both text-to-image generation and multi-step image-to-image editing, controlled by a precision denoise strength handler.
* **Denoising Stream Previews:** Implements a text-to-latent generator iterator loop that updates intermediate image passes dynamically onto the preview canvas before the definitive VAE decode step.
* **Granular Layout Parametrics:** Features expandable settings providing exact modifications over generation seed, steps, resolution configurations (up to 1536px), and LoRA scale blend metrics.
* **Steel Blue Aesthetics:** Crafted using a bespoke developer-focused interface theme wrapped in explicit error feedback pipelines and dynamic execution tracking bars.

### **Repository Structure**

```text
├── app.py
├── LICENSE
├── pre-requirements.txt
├── pyproject.toml
├── README.md
├── requirements.txt
└── uv.lock

```

### **Installation and Requirements**

To configure the FLUX-LoRA-DLC2 suite locally, set up a Python 3.12 environment with the following packages. Ensure your local configuration has access to a dedicated CUDA-capable GPU.

**Standard PIP Installation**

1. Update pip to meet requirements:

```bash
pip install pip>=26.1

```

2. Install standard dependencies:

```bash
pip install -r requirements.txt

```

#### **Running with `uv` (Recommended)**

`uv` is an ultra-fast Python package and project manager written in Rust, which ensures immediate, reproducible execution paths.

**Step 1 — Install `uv**`

* **macOS / Linux:** `curl -LsSf https://astral.sh/uv/install.sh | sh`
* **Windows:** `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`

**Step 2 — Clone the repository**

```bash
git clone https://github.com/PRITHIVSAKTHIUR/FLUX-LoRA-DLC2.git
cd FLUX-LoRA-DLC2

```

**Step 3 — Initialize the project and install dependencies**

```bash
uv sync

```

**Step 4 — Run the script**

```bash
uv run app.py

```

### **Core Requirements List**

The application depends on the following primary libraries (defined in `requirements.txt`):

```text
git+https://github.com/huggingface/accelerate.git
git+https://github.com/huggingface/diffusers.git
huggingface_hub
sentencepiece 
transformers==4.57.6
torchvision
gradio==6.14.0
spaces
torch==2.11.0
numpy
peft

```

---

### **Usage**

Once the FastAPI-backed Gradio client initializes on your device, open your local browser to the host address provided (typically `http://127.0.0.1:7860/`).

1. **Choose a LoRA:** Select an existing entry from the **100+ LoRA DLC's** gallery card grid, or paste an external path into the **Enter Custom LoRA** panel.
2. **Input Prompt:** Provide an explicit description into the primary input box. The system will combine your request with the adapter's specified trigger tokens.
3. **Advanced Settings (Optional):** Expand the Accordion card to tweak the CFG Scale, Image-to-Image denoise values, width, height, or seed properties.
4. **Generate:** Click **Generate** or press enter to see the live step-by-step rendering chain on the image pane.

### **License and Source**

* **License:** [Apache License 2.0](https://github.com/PRITHIVSAKTHIUR/FLUX-LoRA-DLC2/blob/main/LICENSE)
* **GitHub Repository:** [https://github.com/PRITHIVSAKTHIUR/FLUX-LoRA-DLC2](https://github.com/PRITHIVSAKTHIUR/FLUX-LoRA-DLC2)
