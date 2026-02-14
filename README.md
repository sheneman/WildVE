# WildVE: Wildlife Video Extractor

```
 __        __ _  _      _ __     __ _____
 \ \      / /(_)| |  __| |\ \   / /| ____|
  \ \ /\ / / | || | / _` | \ \ / / |  _|
   \ V  V /  | || || (_| |  \ V /  | |___
    \_/\_/   |_||_| \__,_|   \_/   |_____|

     W i l d l i f e   V i d e o   E x t r a c t o r
```

WildVE scans directories of video files for wildlife using an ensemble of AI models, automatically extracting clips containing detected animals and generating summary reports.

## Architecture & Efficiency

WildVE is designed around the principle that in video-based AI pipelines, the dominant bottleneck is disk I/O and CPU-bound video decoding, not GPU inference. Decoding compressed video frames from disk, demuxing, and converting pixel formats are inherently serial, CPU-intensive operations that dwarf the latency of a single forward pass through a neural network. WildVE exploits this asymmetry by keeping all ensemble models memory-resident on the GPU for the entire duration of processing. Once loaded, model weights persist in GPU VRAM across all frames and all videos in a batch, completely eliminating the overhead of repeated model instantiation, weight deserialization, and host-to-device transfers. Each decoded frame is transferred to the GPU once and then routed through all *k* enabled models in sequence, amortizing the cost of the CPU decode and PCIe transfer across the full ensemble. This architecture means that adding additional models to the ensemble incurs only marginal GPU compute cost per frame, while the per-frame I/O and decode cost remains constant regardless of ensemble size. The result is a system where ensemble breadth comes nearly for free relative to the fixed cost of reading and decoding the video stream.

## Ensemble Model Approach

WildVE uses six AI models in an ensemble to maximize detection accuracy and minimize false negatives:

| Model | Type | Purpose |
|-------|------|---------|
| **MegaDetector V5** | Object Detection | General wildlife detection (PytorchWildlife) |
| **MegaDetector V6 (YOLOv9)** | Object Detection | Next-gen wildlife detection |
| **MegaDetector V6 (YOLOv10)** | Object Detection | Next-gen wildlife detection (alternate backbone) |
| **Custom YOLOv8 + EnlightenGAN** | Object Detection | Specialized model trained on enhanced low-light imagery |
| **Florence-2** | Vision-Language | Microsoft's multimodal model for open-vocabulary object detection |
| **CLIP** | Vision-Language | OpenAI's contrastive model for zero-shot image classification |

A frame is flagged as a positive detection when **2 or more** models agree (configurable via `--threshold`), reducing false positives while maintaining high recall. Overall confidence is computed as the average confidence of the agreeing models.

By default, the tiger/EnlightenGAN model is excluded from the ensemble. Use `--all-models` to include it, or `--models` to select specific models.

## Installation

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/sheneman/wildve.git
cd wildve
uv sync
source .venv/bin/activate
```

**Note:** A CUDA-capable GPU is strongly recommended. The ensemble of six models requires significant GPU memory (16GB+ recommended).

## Usage

```bash
python wildve.py <INPUT_DIR> <OUTPUT_DIR> [options]
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `INPUT_DIR` | Directory containing MP4 video files | `inputs` |
| `OUTPUT_DIR` | Directory for extracted clips and metadata | `outputs` |

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `-i`, `--interval` | Seconds between AI sampling frames | `1.0` |
| `-p`, `--padding` | Seconds of video to pad before/after detections | `5.0` |
| `-r`, `--report` | Output report CSV filename | `report.csv` |
| `-j`, `--jobs` | Number of concurrent parallel processes | `4` |
| `-l`, `--logging` | Directory for telemetry log files | `logs` |
| `-n`, `--nobar` | Disable progress bar | `False` |
| `--allframes` | Analyze every frame and output a detailed per-frame CSV report instead of extracting clips | `False` |
| `--models` | Comma-separated list of models to use (options: `md5,md6v9,md6v10,tiger,florence,clip`) | `md5,md6v9,md6v10,florence,clip` |
| `--all-models` | Use all models including the tiger/EnlightenGAN model | `False` |
| `--threshold` | Minimum number of models that must agree for a positive detection | `2` |
| `-g`, `--gpu` | Use GPU if available (default) | `True` |
| `-c`, `--cpu` | Force CPU-only mode | `False` |

### Examples

```bash
# Basic usage
python wildve.py videos/ clips/

# Scan every 2 seconds with 10s padding, using 2 parallel workers
python wildve.py videos/ clips/ -i 2.0 -p 10.0 -j 2

# CPU-only mode (much slower)
python wildve.py videos/ clips/ --cpu

# Disable progress bar (useful for batch/HPC jobs)
python wildve.py videos/ clips/ --nobar

# Analyze every frame (no clip extraction, detailed CSV report)
python wildve.py videos/ results/ --allframes

# Use only MegaDetector models with threshold of 1
python wildve.py videos/ clips/ --models md5,md6v9,md6v10 --threshold 1

# Use all models including the tiger/EnlightenGAN model
python wildve.py videos/ clips/ --all-models
```

## Output

- **Extracted clips**: MP4 files in the output directory, named `<original>_NNN.mp4`
- **Report CSV**: Summary of all extracted clips with start/end frames, times, and confidence statistics
- **Telemetry logs**: Per-process CSV files with per-frame ensemble detection results

## Post-Processing

Use `postprocess.py` to re-encode extracted clips (e.g., for compatibility or compression):

```bash
python postprocess.py
```

Edit the `DIRECTORY_PATH`, `OUTPUT_DIR`, and `NUM_PROCESSES` variables in the script as needed.

## GPU / HPC Notes

- Each parallel process (`-j`) loads its own copy of all enabled models into GPU memory. Reduce `-j` or use fewer `--models` if you encounter out-of-memory errors.
- WildVE will report GPU memory status at startup when running in GPU mode.
- For SLURM-based HPC clusters, request a GPU node and set `-j` based on available GPU memory.
- The `--nobar` flag is recommended for non-interactive batch jobs.

## Model Weights

Most model weights are downloaded automatically on first run:
- **MegaDetector V5, V6**: Downloaded automatically by PytorchWildlife
- **Florence-2**: Downloaded automatically from HuggingFace
- **CLIP**: Downloaded automatically by OpenCLIP

The **tiger/EnlightenGAN YOLOv8** model weights (`best_enlightengan_and_yolov8.pt`) are downloaded automatically from Google Drive when the tiger model is enabled (via `--all-models` or `--models tiger,...`).

## References & Citations

### MegaDetector V5
- Beery, S., Morris, D., & Yang, S. (2019). "Efficient Pipeline for Camera Trap Image Review." *arXiv preprint* [arXiv:1907.06772](https://arxiv.org/abs/1907.06772)
- Microsoft AI for Earth. [CameraTraps](https://github.com/microsoft/CameraTraps)

### MegaDetector V6 (via PytorchWildlife)
- Hernandez, A., et al. (2024). "PytorchWildlife: A Collaborative Deep Learning Framework for Conservation." *arXiv preprint* [arXiv:2405.12930](https://arxiv.org/abs/2405.12930)
- GitHub: [microsoft/CameraTraps](https://github.com/microsoft/CameraTraps) / [PytorchWildlife](https://github.com/microsoft/CameraTraps/blob/main/INSTALLATION.md)

### Florence-2
- Xiao, B., et al. (2024). "Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks." *arXiv preprint* [arXiv:2311.06242](https://arxiv.org/abs/2311.06242)
- HuggingFace: [microsoft/Florence-2-large](https://huggingface.co/microsoft/Florence-2-large)

### CLIP (ViT-B/32)
- Radford, A., et al. (2021). "Learning Transferable Visual Models From Natural Language Supervision." *arXiv preprint* [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)
- GitHub: [openai/CLIP](https://github.com/openai/CLIP), [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip)

### Tiger/EnlightenGAN YOLOv8
- Jiang, Y., et al. (2021). "EnlightenGAN: Deep Light Enhancement without Paired Supervision." *IEEE Transactions on Image Processing*, 30, 2340-2349. [arXiv:1906.06972](https://arxiv.org/abs/1906.06972)
- GitHub: [VITA-Group/EnlightenGAN](https://github.com/VITA-Group/EnlightenGAN)
- Tiger detection pipeline: [Gaurav0502/tiger-detection-using-enlightengan-and-yolo](https://github.com/Gaurav0502/tiger-detection-using-enlightengan-and-yolo)

## WildVE Credits

**Luke Sheneman, Ph.D.**
Director, UI Research Computing & Data Services (RCDS)
UI Institute for Interdisciplinary Data Sciences (IIDS)
https://hpc.uidaho.edu
sheneman@uidaho.edu
GitHub: [sheneman](https://github.com/sheneman)

**Amy Zuckerwise**
Ph.D. Candidate in the Conservation & Coexistence Group
University of Michigan School for Environment and Sustainability
ameliaz@umich.edu
GitHub: [azuckerwise](https://github.com/azuckerwise)

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
