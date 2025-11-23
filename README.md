

# **PLY to SPLATV Converter**

🌟 A tool to create 4D Gaussian Splatting (SPLATV) animations from multiple static 3D Gaussian Splatting PLY files **without training**.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)

## Overview

Convert multiple 3DGS PLY files into an animated SPLATV format that can be viewed in real-time in web browsers. Unlike traditional Spacetime Gaussian Splatting workflows that require multi-camera video training, this tool works directly with pre-rendered PLY frames.

**Key Innovation:** Generate time-axis parameters (trbf_center, trbf_scale) without camera information or neural network training.

## Features

✅ **No Training Required** - Direct conversion from static PLY files  
✅ **No Camera Information** - Works with frame sequences alone  
✅ **Custom Time Control** - Adjust frame timing and duration  
✅ **Importance Sorting** - Optimizes rendering quality  
✅ **Flexible Integration** - Works with any 3DGS PLY files  

## Installation

```bash
pip install numpy plyfile tqdm
```

## Quick Start

### Basic Usage

```bash
# Convert frame sequence to SPLATV
python ply_to_splatv2.py -i ./frames -o output.splatv
```

Your PLY files should be named sequentially (e.g., `frame_000.ply`, `frame_001.ply`, ...).

### View the Result

Upload `output.splatv` to your web server and open in [splatv viewer](https://splatv.vercel.app):

```
https://splatv.vercel.app/?url=YOUR_FILE_URL
```

## Advanced Usage

### Custom Frame Timing

```bash
# Non-uniform time distribution (first frame longer)
python ply_to_splatv2.py -i ./frames -o output.splatv \
  --frame_times "0,0.5,0.75,1.0"
```

### Adjust Temporal Smoothness

```bash
# Sharp transitions (slideshow-like)
python ply_to_splatv2.py -i ./frames -o output.splatv \
  --temporal_sigma 0.1

# Smooth blending
python ply_to_splatv2.py -i ./frames -o output.splatv \
  --temporal_sigma 0.8
```

### Disable Importance Sorting

```bash
python ply_to_splatv2.py -i ./frames -o output.splatv \
  --no_sort
```

## Examples

### Fireworks Animation

[🎆 Live Demo](https://splatv.vercel.app/?url=hanabi2.splatv#[0.48,-0.86,-0.14,0,0.88,0.46,0.12,0,-0.04,-0.18,0.98,0,-0.24,0.28,5.2,1])

```bash
# Generate fireworks frames
python examples/fireworks_generator.py

# Convert to SPLATV
!python ply_to_splatv2.py -i ./frames -o hanabi2.splatv --frame_times "0.0,0.3,0.7,1.0,1.3,1.8"

# Result: 6-stage fireworks animation!
```

### Other Use Cases

- **Physics Simulations** → Animate simulation frames
- **3DGS Frame Sequences** → Create smooth videos
- **Scene Transitions** → Morph between different scenes
- **Stop-Motion Style** → Frame-by-frame 3DGS animation

## How It Works

### Standard Spacetime Gaussian Workflow
```
Multi-camera video + Camera poses
    ↓ (Training with SpacetimeGaussians)
Trained model with motion parameters
    ↓
SPLATV file
```

### This Tool's Workflow
```
Multiple static 3DGS PLY files
    ↓ (Direct conversion - no training)
SPLATV file with synthesized time parameters
```

### Technical Details

The tool generates temporal parameters for each Gaussian:
- `trbf_center`: Time position (0.0-1.0)
- `trbf_scale`: Temporal width (controls visibility duration)
- `motion_*`: Set to 0 (no spatial motion)
- `omega_*`: Set to 0 (no rotation change)

This creates a "slideshow" effect where each frame appears at its designated time.

## File Format

Output SPLATV format:
```
[Magic: 0x674b]
[JSON Length: uint32]
[JSON Metadata: cameras, texture size]
[Binary Texture Data: 64 bytes per Gaussian]
```

Each Gaussian contains:
- Position (xyz)
- Rotation (quaternion)
- Scale
- Color/Opacity
- Temporal parameters (trbf_center, trbf_scale)

## Requirements

- Python 3.7+
- numpy
- plyfile
- tqdm

## Command Line Options

```
usage: ply_to_splatv2.py [-h] -i INPUT_DIR [-p PATTERN] [-o OUTPUT]
                        [--temporal_sigma TEMPORAL_SIGMA]
                        [--frame_times FRAME_TIMES] [--no_sort]

optional arguments:
  -i, --input_dir       Directory containing PLY files
  -p, --pattern         Filename pattern (default: frame_*.ply)
  -o, --output          Output file path (default: output.splatv)
  --temporal_sigma      Temporal Gaussian width (default: 0.3)
  --frame_times         Custom time array (e.g., "0,0.3,0.7,1.0")
  --no_sort            Disable importance sorting
```

## Viewer Recommendations

### Official splaTV
- URL: https://splatv.vercel.app

### Improved Fork
For one-way looping with pause control:
- Repository: [tztechno/splatv](https://github.com/tztechno/splatv)
- Features: Adjustable pause duration, one-way animation

## Limitations

- No learned motion blur
- No view-dependent temporal effects
- Discrete frame switching (not continuous motion)
- Larger file size than single PLY

## Related Projects

- [antimatter15/splaTV](https://github.com/antimatter15/splaTV) - Original web viewer
- [OPPO SpacetimeGaussians](https://github.com/oppo-us-research/SpacetimeGaussians) - Training-based approach
- [macaburguera/splaTV](https://github.com/macaburguera/splaTV) - Trained model converter

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{ply_to_splatv,
  author = {stpete ishii},
  title = {PLY to SPLATV Converter: Training-Free 4D Gaussian Splatting},
  year = {2025},
  url = {https://github.com/tztechno/ply-to-splatv}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by [antimatter15/splaTV](https://github.com/antimatter15/splaTV)
- Based on Spacetime Gaussian Splatting format
- Thanks to the 3DGS community

---

**Note:** This is an experimental tool. For production use with learned motion, consider training with [SpacetimeGaussians](https://github.com/oppo-us-research/SpacetimeGaussians).

🎆 Happy Splatting!



## Version Information

### Current Version: 2.0.0

**Recommended:** Use `ply_to_splatv2.py` (v2) for all new projects.
```bash
# Latest version (recommended)
python ply_to_splatv2.py -i ./frames -o output.splatv

# Legacy version (simple, no sorting)
python ply_to_splatv1.py -i ./frames -o output.splatv
```

See [CHANGELOG.md](CHANGELOG.md) for detailed version history.

### Version Comparison

| Feature | v1.0 (Legacy) | v2.0 (Current) |
|---------|---------------|----------------|
| Conversion speed | Baseline | 🚀 3x faster |
| Rendering quality | Standard | ⭐ Optimized |
| Custom timing | ❌ | ✅ |
| Importance sorting | ❌ | ✅ |
| Use case | Learning/Simple | Production |

```
