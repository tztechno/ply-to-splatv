

#!/usr/bin/env python3

"""
PLY to SPLATV Converter - Version 2.0
複数の3DGS PLYファイルをsplaTV viewer用のsplatvファイルに変換

Version 2.0 Features:
- Importance-based sorting for optimal rendering
- Custom frame timing support
- Fast NumPy-based packHalf2x16
- Detailed progress logging

Version History:
- v1.0: Initial release with basic conversion
- v2.0: Added sorting, custom timing, performance improvements

使用方法:
    python ply_to_splatv.py -i ./frames -o output.splatv --temporal_sigma 0.3

"""

import numpy as np
from plyfile import PlyData
import glob
import os
import argparse
import json
import struct
from tqdm import tqdm


def float_to_half(f):
    """Converts float32 to float16 (fast version)"""
    return np.float16(f).view(np.uint16)


def pack_half2x16_batch(arr):
    """
    Batch conversion of float32 array to half2x16 (fast version)
    arr: array of shape (n, 2) or (2n,)
    """
    arr = np.asarray(arr, dtype=np.float32)
    
    if len(arr.shape) == 1:
        # For 1D array
        assert arr.shape[0] % 2 == 0
        x = arr[0::2]
        y = arr[1::2]
    else:
        # For 2D array
        x = arr[:, 0]
        y = arr[:, 1]
    
    n = x.shape[0]
    uint32_data = np.ndarray((n,), dtype=np.uint32)
    
    # Create Float16 view
    f16_data = np.lib.stride_tricks.as_strided(
        uint32_data.view(dtype=np.float16),
        shape=(2, n),
        strides=(1 * 2, 2 * 2),
        writeable=True,
    )
    
    f16_data[0] = x
    f16_data[1] = y
    
    return uint32_data


def pack_half2x16(x, y):
    """Packs two floats into half2x16 (individual version)"""
    hx = int(np.float16(x).view(np.uint16))
    hy = int(np.float16(y).view(np.uint16))
    return (hx | (hy << 16)) & 0xFFFFFFFF


def load_3dgs_ply(filepath):
    """Loads a standard 3DGS format PLY file"""
    plydata = PlyData.read(filepath)
    vertex = plydata['vertex']
    
    xyz = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1).astype(np.float32)
    opacity = vertex['opacity'].astype(np.float32)
    
    scale = np.stack([
        vertex['scale_0'], 
        vertex['scale_1'], 
        vertex['scale_2']
    ], axis=1).astype(np.float32)
    
    rotation = np.stack([
        vertex['rot_0'], 
        vertex['rot_1'], 
        vertex['rot_2'], 
        vertex['rot_3']
    ], axis=1).astype(np.float32)
    
    f_dc = np.stack([
        vertex['f_dc_0'], 
        vertex['f_dc_1'], 
        vertex['f_dc_2']
    ], axis=1).astype(np.float32)
    
    return {
        'xyz': xyz,
        'opacity': opacity,
        'scale': scale,
        'rotation': rotation,
        'f_dc': f_dc,
        'n_points': xyz.shape[0]
    }


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sh_to_rgb(f_dc):
    """Converts SH DC coefficients to RGB (0-255)"""
    C0 = 0.28209479177387814
    rgb = 0.5 + C0 * f_dc
    rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    return rgb


def build_splatv_texture(all_data, n_frames, temporal_sigma=0.3, frame_times=None, sort_by_importance=True):
    """
    Builds the splatv texture from multi-frame data.
    
    Each Gaussian: 16 x uint32 = 64 bytes
    
    Args:
        all_data: List of frame data
        n_frames: Number of frames
        temporal_sigma: Temporal Gaussian width
        frame_times: Custom time array (None=uniform placement)
        sort_by_importance: Sorts by importance (recommended True)
    """
    total_points = sum(d['n_points'] for d in all_data)
    
    # Calculate texture size
    texwidth = 4096
    texheight = int(np.ceil((4 * total_points) / texwidth))
    
    # Texture data (uint32)
    texdata = np.zeros(texwidth * texheight * 4, dtype=np.uint32)
    texdata_f = texdata.view(np.float32)
    texdata_u8 = texdata.view(np.uint8)
    
    # Calculate frame times
    if frame_times is None:
        # Default: uniform placement
        if n_frames > 1:
            frame_times = np.linspace(0.0, 1.0, n_frames)
        else:
            frame_times = np.array([0.5])
    else:
        frame_times = np.array(frame_times)
        # Normalize to 0.0-1.0
        frame_times = (frame_times - frame_times.min()) / (frame_times.max() - frame_times.min())
    
    # Frame interval (for sigma calculation)
    if n_frames > 1:
        avg_interval = np.mean(np.diff(frame_times))
    else:
        avg_interval = 1.0
    
    sigma = temporal_sigma * avg_interval
    
    print(f"\nTexture size: {texwidth} x {texheight}")
    print(f"Frame times: {frame_times}")
    print(f"Temporal sigma: {sigma:.6f}")
    print(f"Importance sorting: {sort_by_importance}")
    
    # Preparation for importance sorting
    if sort_by_importance:
        print("\nCalculating importance...")
        importance_list = []
        point_data = []
        
        for frame_idx, data in enumerate(tqdm(all_data, desc="Calculating importance")):
            trbf_center = frame_times[frame_idx]
            n_points = data['n_points']
            
            # Importance = opacity * volume
            importance = (
                sigmoid(data['opacity']) * np.exp(data['scale'][:, 0]) * np.exp(data['scale'][:, 1]) * np.exp(data['scale'][:, 2])
            )
            
            for i in range(n_points):
                importance_list.append(importance[i])
                point_data.append({
                    'frame_idx': frame_idx,
                    'point_idx': i,
                    'trbf_center': trbf_center,
                    'data': data
                })
        
        # Sort by importance (descending)
        sorted_indices = np.argsort(-np.array(importance_list))
        print(f"Sorting complete: {len(sorted_indices)} points")
    else:
        # No sorting
        point_data = []
        for frame_idx, data in enumerate(all_data):
            trbf_center = frame_times[frame_idx]
            for i in range(data['n_points']):
                point_data.append({
                    'frame_idx': frame_idx,
                    'point_idx': i,
                    'trbf_center': trbf_center,
                    'data': data
                })
        sorted_indices = np.arange(len(point_data))
    
    # Building texture
    print("\nBuilding texture...")
    exp_trbf_scale = sigma
    
    for j in tqdm(range(len(sorted_indices)), desc="Building texture"):
        idx = sorted_indices[j]
        pd = point_data[idx]
        data = pd['data']
        i = pd['point_idx']
        trbf_center = pd['trbf_center']
        
        # [0-2]: x, y, z (float32)
        texdata_f[16 * j + 0] = data['xyz'][i, 0]
        texdata_f[16 * j + 1] = data['xyz'][i, 1]
        texdata_f[16 * j + 2] = data['xyz'][i, 2]
        
        # [3-6]: rotation & scale (half2x16)
        texdata[16 * j + 3] = pack_half2x16(data['rotation'][i, 0], data['rotation'][i, 1])
        texdata[16 * j + 4] = pack_half2x16(data['rotation'][i, 2], data['rotation'][i, 3])
        texdata[16 * j + 5] = pack_half2x16(np.exp(data['scale'][i, 0]), np.exp(data['scale'][i, 1]))
        texdata[16 * j + 6] = pack_half2x16(np.exp(data['scale'][i, 2]), 0)
        
        # [7]: r, g, b, opacity (uint8 x 4)
        rgb = sh_to_rgb(data['f_dc'][i:i+1])[0]
        alpha = int(sigmoid(data['opacity'][i]) * 255)
        texdata_u8[4 * (16 * j + 7) + 0] = rgb[0]
        texdata_u8[4 * (16 * j + 7) + 1] = rgb[1]
        texdata_u8[4 * (16 * j + 7) + 2] = rgb[2]
        texdata_u8[4 * (16 * j + 7) + 3] = alpha
        
        # [8-14]: motion & omega (all 0)
        for k in range(8, 15):
            texdata[16 * j + k] = 0
        
        # [15]: trbf_center, exp(trbf_scale)
        texdata[16 * j + 15] = pack_half2x16(trbf_center, exp_trbf_scale)
    
    return texdata, texwidth, texheight, total_points


def create_default_cameras():
    """Default camera settings (same format as a standard splatv file)"""
    return [{
        "id": 0,
        "img_name": "camera_0001",
        "width": 1920,
        "height": 1080,
        "position": [0.0, 0.0, 3.0],
        "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "fy": 1000.0,
        "fx": 1000.0
    }]


def write_splatv(output_path, texdata, texwidth, texheight, cameras=None):
    """Writes the splatv file"""
    
    if cameras is None:
        cameras = create_default_cameras()
    
    # Metadata JSON (no spaces, same format as a standard splatv file)
    metadata = [{
        "type": "splat",
        "size": texdata.nbytes,
        "texwidth": texwidth,
        "texheight": texheight,
        "cameras": cameras
    }]
    
    json_bytes = json.dumps(metadata, separators=(',', ':')).encode('utf-8')
    
    print(f"JSON length: {len(json_bytes)}")
    print(f"JSON: {json_bytes[:100]}...")
    
    # Header
    magic = struct.pack('<I', 0x674b)  # "Kg" (little-endian)
    json_len = struct.pack('<I', len(json_bytes))
    
    # File writing
    with open(output_path, 'wb') as f:
        f.write(magic)
        f.write(json_len)
        f.write(json_bytes)
        f.write(texdata.tobytes())
    
    return output_path


def convert_frames_to_splatv(ply_files, output_path, temporal_sigma=0.3, cameras=None, frame_times=None, sort_by_importance=True):
    """
    Converts multiple PLY files into a single splatv file
    
    Args:
        ply_files: List of PLY file paths
        output_path: Output file path
        temporal_sigma: Temporal Gaussian width
        cameras: Camera settings (None=default)
        frame_times: Array of times for each frame (None=uniform placement)
                     Example: [0, 0.2, 0.5, 1.0] - keeps the first frame displayed longer
        sort_by_importance: Sorts by importance (recommended)
    """
    
    n_frames = len(ply_files)
    print(f"=" * 60)
    print(f"PLY to SPLATV Converter")
    print(f"=" * 60)
    print(f"Number of frames: {n_frames}")
    print(f"Temporal sigma: {temporal_sigma}")
    
    # Load all frames
    print(f"\n[1/3] Loading PLY files...")
    all_data = []
    for filepath in tqdm(ply_files):
        data = load_3dgs_ply(filepath)
        all_data.append(data)
    
    total_points = sum(d['n_points'] for d in all_data)
    print(f"Total number of Gaussians: {total_points:,}")
    
    # Build texture
    print(f"\n[2/3] Building texture data...")
    texdata, texwidth, texheight, _ = build_splatv_texture(
        all_data, n_frames, temporal_sigma, frame_times, sort_by_importance
    )
    
    # Write splatv file
    print(f"\n[3/3] Writing splatv file...")
    write_splatv(output_path, texdata, texwidth, texheight, cameras)
    
    file_size = os.path.getsize(output_path)
    print(f"\n" + "=" * 60)
    print(f"DONE!")
    print(f"=" * 60)
    print(f"Output file: {output_path}")
    print(f"File size: {file_size / 1024 / 1024:.2f} MB")
    print(f"Total frames: {n_frames}")
    print(f"Total Gaussians: {total_points:,}")
    print(f"Texture: {texwidth} x {texheight}")
    print(f"\nOpen with splatv viewer:")
    print(f"  https://splatv.vercel.app/?url=YOUR_FILE_URL")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Converts multiple 3DGS PLY files into the splatv format.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python ply_to_splatv.py -i ./frames -o output.splatv
  python ply_to_splatv.py -i ./frames -o output.splatv --temporal_sigma 0.2

About temporal_sigma:
  - 0.1-0.2: Sharp transition (slideshow-like)
  - 0.3-0.5: Standard transition
  - 1.0+: Smooth blending
"""
    )
    parser.add_argument('-i', '--input_dir', type=str, required=True,
                        help='Directory containing the PLY files.')
    parser.add_argument('-p', '--pattern', type=str, default='frame_*.ply',
                        help='Filename pattern (default: frame_*.ply).')
    parser.add_argument('-o', '--output', type=str, default='output.splatv',
                        help='Output file path (default: output.splatv).')
    parser.add_argument('--temporal_sigma', type=float, default=0.3,
                        help='Temporal Gaussian width (default: 0.3).')
    parser.add_argument('--frame_times', type=str, default=None,
                        help='Custom time array (comma-separated). Example: "0,0.3,0.6,1.0".')
    parser.add_argument('--no_sort', action='store_true',
                        help='Disable importance sorting (not recommended usually).')
    
    args = parser.parse_args()
    
    # Search for PLY files
    pattern = os.path.join(args.input_dir, args.pattern)
    ply_files = sorted(glob.glob(pattern))
    
    if not ply_files:
        print(f"Error: No files found matching pattern '{pattern}'")
        return
    
    print(f"Detected PLY files: {len(ply_files)}")
    print(f"First: {os.path.basename(ply_files[0])}")
    print(f"Last: {os.path.basename(ply_files[-1])}")
    
    # Parse custom time array
    frame_times = None
    if args.frame_times:
        frame_times = [float(x.strip()) for x in args.frame_times.split(',')]
        if len(frame_times) != len(ply_files):
            print(f"Error: The number of frame_times ({len(frame_times)}) does not match the number of files ({len(ply_files)})")
            return
        print(f"Custom times: {frame_times}")
    
    convert_frames_to_splatv(
        ply_files=ply_files,
        output_path=args.output,
        temporal_sigma=args.temporal_sigma,
        frame_times=frame_times,
        sort_by_importance=not args.no_sort
    )


if __name__ == '__main__':
    main()

