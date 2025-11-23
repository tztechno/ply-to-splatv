# ply-to-splatv

```

!python ply_to_splatv2.py -i ./frames -o hanabi2.splatv --frame_times "0.0,0.3,0.7,1.0,1.3,1.8"
```

## Version Information

### Current Version: 2.0.0

**Recommended:** Use `ply_to_splatv.py` (v2) for all new projects.
```bash
# Latest version (recommended)
python ply_to_splatv2.py -i ./frames -o output.splatv

# Legacy version (simple, no sorting)
python ply_to_splatv1.py -i ./frames -o output.splatv
```

See [CHANGELOG.md](CHANGELOG.md) for detailed version history and [migration guide](docs/migration_v1_to_v2.md).

### Version Comparison

| Feature | v1.0 (Legacy) | v2.0 (Current) |
|---------|---------------|----------------|
| Conversion speed | Baseline | 🚀 3x faster |
| Rendering quality | Standard | ⭐ Optimized |
| Custom timing | ❌ | ✅ |
| Importance sorting | ❌ | ✅ |
| Use case | Learning/Simple | Production |

```
