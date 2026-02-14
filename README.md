# WildVE: Wildlife Video Extractor

WildVE scans directories of video files for wildlife using an ensemble of AI models, automatically extracting clips containing detected animals and generating summary reports.

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

A frame is flagged as a positive detection when **2 or more** models agree, reducing false positives while maintaining high recall. Overall confidence is computed as a weighted average adjusted by the fraction of agreeing models.

## Installation

```bash
git clone https://github.com/sheneman/wildve.git
cd wildve
pip install -r requirements.txt
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

- Each parallel process (`-j`) loads its own copy of all six models into GPU memory. Reduce `-j` if you encounter out-of-memory errors.
- WildVE will report GPU memory status at startup when running in GPU mode.
- For SLURM-based HPC clusters, request a GPU node and set `-j` based on available GPU memory.
- The `--nobar` flag is recommended for non-interactive batch jobs.

## Custom Model

The custom YOLOv8 model (`best_enlightengan_and_yolov8.pt`) is trained on imagery enhanced with EnlightenGAN for improved low-light detection. This model file must be present in the working directory.

## License

MIT License. See [LICENSE](LICENSE) for details.
