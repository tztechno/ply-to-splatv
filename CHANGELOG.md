# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [2.0.0] - 2025-11-23

### Added
- **Importance-based sorting** - Gaussians sorted by opacity × volume for better rendering quality
- **Custom frame timing** - `--frame_times` parameter for non-uniform time distribution
- **Optimized packHalf2x16** - Fast NumPy stride-based implementation
- **Detailed progress logging** - Multi-stage progress bars with tqdm
- **Automatic time normalization** - Frame times auto-normalized to 0.0-1.0 range
- **Sorting control** - `--no_sort` flag to disable importance sorting

### Changed
- Default output extension: `.ply` → `.splatv`
- Enhanced JSON formatting (no spaces, compact)
- Improved error messages and validation
- Better handling of edge cases (single frame, etc.)

### Performance
- ~3x faster for large datasets (>10k Gaussians per frame)
- Reduced memory usage with batch processing

### Technical Details
- Adopted NumPy stride tricks for half-precision conversion (inspired by macaburguera implementation)
- Frame time calculation now supports both uniform and custom distributions
- Added comprehensive type hints and documentation

## [1.0.0] - 2025-11-23

### Initial Release
- Basic PLY to SPLATV conversion
- Multiple frame support
- Temporal sigma control
- Standard 3DGS PLY format support
- Camera metadata generation

### Features
- Convert multiple PLY files to single SPLATV
- Automatic trbf_center/trbf_scale generation
- Motion/omega parameters set to 0 (static frames)
- Command-line interface with argparse

---

## Version Comparison

| Feature | v1.0 | v2.0 |
|---------|------|------|
| Basic conversion | ✅ | ✅ |
| Temporal sigma | ✅ | ✅ |
| Custom frame times | ❌ | ✅ |
| Importance sorting | ❌ | ✅ |
| Fast packHalf2x16 | ❌ | ✅ |
| Progress bars | Basic | Detailed |
| Performance | Baseline | 3x faster |

## Migration Guide (v1 → v2)

### Command Compatibility
All v1 commands work in v2:
```bash
# v1 command
python ply_to_splatv1.py -i ./frames -o output.splatv

# v2 equivalent (identical behavior)
python ply_to_splatv2.py -i ./frames -o output.splatv
```

### New Features in v2
```bash
# Custom timing (v2 only)
python ply_to_splatv2.py -i ./frames -o output.splatv \
  --frame_times "0,0.3,0.7,1.0"

# Disable sorting (v2 only)
python ply_to_splatv2.py -i ./frames -o output.splatv \
  --no_sort
```

### When to Use v1
- Simple projects with few frames
- Debugging without sorting complexity
- Educational purposes (simpler code)

### When to Use v2 (Recommended)
- Production use
- Large datasets (>5k Gaussians/frame)
- Need custom timing control
- Want best rendering quality

---

## Roadmap

### Planned for v2.1
- [ ] Batch processing of multiple sequences
- [ ] GPU acceleration option
- [ ] Built-in preview generation
- [ ] Support for extended SH coefficients

### Future Considerations
- [ ] WebAssembly version for browser use
- [ ] Integration with 3DGS training pipelines
- [ ] Compression/streaming support
- [ ] Motion blur synthesis

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## Issues & Bugs

Please report issues on [GitHub Issues](https://github.com/yourusername/ply-to-splatv/issues).
