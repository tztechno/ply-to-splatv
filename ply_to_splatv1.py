

#!/usr/bin/env python3
"""
PLY to SPLATV Converter
Converts multiple 3DGS PLY files to the splatv format.

splatv format:
  - Header: [magic 4B][json_len 4B][json][binary_data]
  - Each Gaussian: 64 bytes (16 x uint32)

Usage:
    python ply_to_splatv_converter.py -i ./frames -o output.splatv

Required libraries:
    pip install numpy plyfile tqdm
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
    """Converts float32 to float16"""
    return int(np.float16(f).view(np.uint16))


def pack_half2x16(x, y):
    """Packs two floats into half2x16"""
    hx = float_to_half(x)
    hy = float_to_half(y)
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


def build_splatv_texture(all_data, n_frames, temporal_sigma=0.3, frame_times=None):
    """
    Builds the splatv texture from multi-frame data.
    
    Each Gaussian: 16 x uint32 = 64 bytes
    
    Args:
        all_data: List of frame data
        n_frames: Number of frames
        temporal_sigma: Temporal Gaussian width
        frame_times: Custom time array (None=evenly spaced)
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
        # Default: evenly spaced
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
    
    j = 0  # Output index
    
    for frame_idx, data in enumerate(tqdm(all_data, desc="Building texture")):
        # trbf_center
        trbf_center = frame_times[frame_idx]
        
        # Store exp(trbf_scale) directly
        exp_trbf_scale = sigma
        
        n_points = data['n_points']
        xyz = data['xyz']
        rot = data['rotation']
        scale = data['scale']
        f_dc = data['f_dc']
        opacity = data['opacity']
        
        # RGB conversion
        rgb = sh_to_rgb(f_dc)
        
        # Opacity conversion (sigmoid)
        alpha = (sigmoid(opacity) * 255).astype(np.uint8)
        
        for i in range(n_points):
            # [0-2]: x, y, z (float32)
            texdata_f[16 * j + 0] = xyz[i, 0]
            texdata_f[16 * j + 1] = xyz[i, 1]
            texdata_f[16 * j + 2] = xyz[i, 2]
            
            # [3]: rot_0, rot_1 (half2x16)
            texdata[16 * j + 3] = pack_half2x16(rot[i, 0], rot[i, 1])
            
            # [4]: rot_2, rot_3 (half2x16)
            texdata[16 * j + 4] = pack_half2x16(rot[i, 2], rot[i, 3])
            
            # [5]: exp(scale_0), exp(scale_1) (half2x16)
            texdata[16 * j + 5] = pack_half2x16(np.exp(scale[i, 0]), np.exp(scale[i, 1]))
            
            # [6]: exp(scale_2), 0 (half2x16)
            texdata[16 * j + 6] = pack_half2x16(np.exp(scale[i, 2]), 0)
            
            # [7]: r, g, b, opacity (uint8 x 4)
            texdata_u8[4 * (16 * j + 7) + 0] = rgb[i, 0]
            texdata_u8[4 * (16 * j + 7) + 1] = rgb[i, 1]
            texdata_u8[4 * (16 * j + 7) + 2] = rgb[i, 2]
            texdata_u8[4 * (16 * j + 7) + 3] = alpha[i]
            
            # [8-12]: motion_0-8 (all 0, no motion)
            texdata[16 * j + 8] = pack_half2x16(0, 0)   # motion_0, motion_1
            texdata[16 * j + 9] = pack_half2x16(0, 0)   # motion_2, motion_3
            texdata[16 * j + 10] = pack_half2x16(0, 0)  # motion_4, motion_5
            texdata[16 * j + 11] = pack_half2x16(0, 0)  # motion_6, motion_7
            texdata[16 * j + 12] = pack_half2x16(0, 0)  # motion_8, 0
            
            # [13-14]: omega_0-3 (all 0, no rotation change)
            texdata[16 * j + 13] = pack_half2x16(0, 0)  # omega_0, omega_1
            texdata[16 * j + 14] = pack_half2x16(0, 0)  # omega_2, omega_3
            
            # [15]: trbf_center, exp(trbf_scale) (half2x16)
            texdata[16 * j + 15] = pack_half2x16(trbf_center, exp_trbf_scale)
            
            j += 1
    
    return texdata, texwidth, texheight, total_points


def create_default_cameras():
    """Default camera settings (same format as a normal splatv)"""
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
    
    # Metadata JSON (no spaces, same format as a normal splatv)
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
    magic = struct.pack('<I', 0x674b)  # "Kg" (little endian)
    json_len = struct.pack('<I', len(json_bytes))
    
    # Write file
    with open(output_path, 'wb') as f:
        f.write(magic)
        f.write(json_len)
        f.write(json_bytes)
        f.write(texdata.tobytes())
    
    return output_path


def convert_frames_to_splatv(ply_files, output_path, temporal_sigma=0.3, cameras=None, frame_times=None):
    """
    Converts multiple PLY files into a single splatv file.
    
    Args:
        ply_files: List of PLY file paths
        output_path: Output file path
        temporal_sigma: Temporal Gaussian width
        cameras: Camera settings (None=default)
        frame_times: Array of times for each frame (None=evenly spaced)
                     Example: [0, 0.2, 0.5, 1.0] - first frame lasts longer
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
        all_data, n_frames, temporal_sigma, frame_times
    )
    
    # Write splatv file
    print(f"\n[3/3] Writing splatv file...")
    write_splatv(output_path, texdata, texwidth, texheight, cameras)
    
    file_size = os.path.getsize(output_path)
    print(f"\n" + "=" * 60)
    print(f"Complete!")
    print(f"=" * 60)
    print(f"Output file: {output_path}")
    print(f"File size: {file_size / 1024 / 1024:.2f} MB")
    print(f"Total frames: {n_frames}")
    print(f"Total Gaussians: {total_points:,}")
    print(f"Texture: {texwidth} x {texheight}")
    print(f"\nOpen in splatv viewer:")
    print(f"  https://splatv.vercel.app/?url=YOUR_FILE_URL")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Converts multiple 3DGS PLY files to the splatv format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ply_to_splatv_converter.py -i ./frames -o output.splatv
  python ply_to_splatv_converter.py -i ./frames -o output.splatv --temporal_sigma 0.2

About temporal_sigma:
  - 0.1-0.2: Sharp transitions (slideshow-like)
  - 0.3-0.5: Standard transitions
  - 1.0+: Smooth blend
"""
    )
    parser.add_argument('-i', '--input_dir', type=str, required=True,
                        help='Directory containing the PLY files')
    parser.add_argument('-p', '--pattern', type=str, default='frame_*.ply',
                        help='Filename pattern (default: frame_*.ply)')
    parser.add_argument('-o', '--output', type=str, default='output.splatv',
                        help='Output file path (default: output.splatv)')
    parser.add_argument('--temporal_sigma', type=float, default=0.3,
                        help='Temporal Gaussian width (default: 0.3)')
    parser.add_argument('--frame_times', type=str, default=None,
                        help='Custom time array (comma-separated) e.g., "0,0.3,0.6,1.0"')
    
    args = parser.parse_args()
    
    # Search for PLY files
    pattern = os.path.join(args.input_dir, args.pattern)
    ply_files = sorted(glob.glob(pattern))
    
    if not ply_files:
        print(f"Error: No files matching pattern '{pattern}' found")
        return
    
    print(f"PLY files detected: {len(ply_files)}")
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
        frame_times=frame_times
    )


if __name__ == '__main__':
    main()


