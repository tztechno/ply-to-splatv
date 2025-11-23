import torch
import numpy as np
from plyfile import PlyData, PlyElement
import math
import colorsys

class FireworksSpacetimeModel:
    def __init__(self, num_gaussians=3000):
        self.device = "cpu"
        self.num = num_gaussians
        self.n_p = 2  # Quadratic motion (to represent gravity)
        
        print(f"--- Generating {num_gaussians} Fireworks Gaussians ---")
        
        # Firework types (0: spherical, 1: hemispherical, 2: ring, 3: multi-ring)
        self.firework_type = torch.randint(0, 4, (self.num,))
        
        # 1. Initial position - starting point before launch
        initial_pos = torch.zeros((self.num, 3))
        initial_pos[:, 2] = torch.rand(self.num) * 0.1 - 0.5  # Start near ground
        
        # 2. Motion coefficients (quadratic motion)
        # mu(t) = b0 + b1*t + b2*t^2
        self.motion_coeffs = torch.zeros((self.num, self.n_p + 1, 3))
        
        # b0: Initial position (before launch)
        self.motion_coeffs[:, 0, :] = initial_pos
        
        # b1: Initial velocity (upward)
        launch_speed = 2.0 + torch.rand(self.num) * 1.0
        self.motion_coeffs[:, 1, :] = torch.zeros((self.num, 3))
        self.motion_coeffs[:, 1, 2] = launch_speed  # Mainly upward along Z-axis
        
        # b2: Acceleration (gravity and explosion force)
        gravity = -1.8  # Gravity acceleration
        
        self.motion_coeffs[:, 2, :] = torch.zeros((self.num, 3))
        self.motion_coeffs[:, 2, 2] = gravity  # Gravity
        
        # 3. Explosion timing and characteristics
        self.explosion_time = 0.8 + torch.rand(self.num) * 0.2  # Explode at 0.8-1.0 seconds
        self.explosion_direction = torch.randn(self.num, 3)  # Explosion direction
        self.explosion_direction = self.explosion_direction / self.explosion_direction.norm(dim=1, keepdim=True)
        
        # Set explosion patterns based on firework type
        for i in range(self.num):
            if self.firework_type[i] == 0:  # Spherical
                self.explosion_direction[i] = torch.randn(3)
            elif self.firework_type[i] == 1:  # Hemispherical
                direction = torch.randn(3)
                direction[2] = abs(direction[2])  # Upward hemisphere
                self.explosion_direction[i] = direction
            elif self.firework_type[i] == 2:  # Ring
                theta = torch.rand(1) * 2 * math.pi
                self.explosion_direction[i] = torch.tensor([math.cos(theta), math.sin(theta), 0.0])
            else:  # Multi-ring
                direction = torch.randn(3)
                direction[2] = direction[2] * 0.3  # Mostly planar
                self.explosion_direction[i] = direction
                
        self.explosion_direction = self.explosion_direction / self.explosion_direction.norm(dim=1, keepdim=True)
        
        # 4. Color settings (vibrant colors using HSV color space)
        self.colors = torch.zeros((self.num, 3))
        for i in range(self.num):
            hue = torch.rand(1).item()  # 0.0-1.0
            saturation = 0.8 + torch.rand(1).item() * 0.2  # 0.8-1.0
            value = 0.8 + torch.rand(1).item() * 0.2  # 0.8-1.0
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            self.colors[i] = torch.tensor(rgb)
        
        # 5. Size and opacity
        self.base_scaling = torch.ones((self.num, 3)) * 0.015
        self.size_variation = torch.rand(self.num) * 0.01 + 0.005
        
        # 6. Time decay parameters
        self.fade_start = 1.2  # Fade start time
        self.fade_duration = 0.8  # Fade duration

    def get_splat_at_t(self, t: float):
        """Calculate firework state at specified time t"""
        
        # Pre-explosion launch motion
        if t < self.explosion_time.min().item():
            # Simple projectile motion (during launch)
            xyz = (self.motion_coeffs[:, 0, :] + 
                  self.motion_coeffs[:, 1, :] * t + 
                  self.motion_coeffs[:, 2, :] * t * t)
            
            # Small and dim during launch
            current_scaling = self.base_scaling * 0.3
            current_colors = self.colors * 0.3
            opacity = torch.ones((self.num, 1)) * 0.7
            
        else:
            # Post-explosion motion
            explosion_progress = (t - self.explosion_time) / 0.5
            explosion_progress = torch.clamp(explosion_progress, 0.0, 1.0)
            
            # Fix: Properly handle tensor shapes for explosion time
            explosion_time_expanded = self.explosion_time.unsqueeze(1).expand(-1, 3)
            
            # Base position for explosion (apex of the launch)
            apex_pos = (self.motion_coeffs[:, 0, :] + 
                       self.motion_coeffs[:, 1, :] * explosion_time_expanded +
                       self.motion_coeffs[:, 2, :] * explosion_time_expanded * explosion_time_expanded)
            
            # Explosion spread
            explosion_speed = 1.5 + torch.rand(self.num) * 1.0
            explosion_offset = self.explosion_direction * explosion_speed.unsqueeze(1) * explosion_progress.unsqueeze(1)
            
            # Gravity effect after explosion
            fall_time = t - self.explosion_time
            gravity_effect = torch.zeros((self.num, 3))
            gravity_effect[:, 2] = -0.5 * 1.8 * fall_time * fall_time
            
            xyz = apex_pos + explosion_offset + gravity_effect
            
            # Size and color after explosion
            size_factor = 1.0 - explosion_progress * 0.5
            current_scaling = self.base_scaling * size_factor.unsqueeze(1)
            current_scaling += self.size_variation.unsqueeze(1)
            
            # Brightness variation (twinkling effect)
            brightness = 0.7 + 0.3 * torch.sin(t * 20 + torch.rand(self.num) * math.pi).unsqueeze(1)
            current_colors = self.colors * brightness
            
            # Opacity decay - FIXED
            fade_progress = max(0.0, min(1.0, (t - self.fade_start) / self.fade_duration))
            opacity = torch.ones((self.num, 1)) * (1.0 - fade_progress) * 0.9
        
        # Fixed rotation (identity quaternion)
        rotation = torch.zeros((self.num, 4))
        rotation[:, 0] = 1.0  # w=1

        return {
            "xyz": xyz,
            "rotation": rotation,
            "opacity": opacity,
            "scaling": current_scaling,
            "features": current_colors
        }

def save_ply(splat_dict, filename):
    """Save as PLY format"""
    xyz = splat_dict["xyz"].numpy()
    features = splat_dict["features"].numpy()
    opacity = splat_dict["opacity"].numpy()
    scaling = splat_dict["scaling"].numpy()
    rotation = splat_dict["rotation"].numpy()
    
    # Opacity logit transformation
    op_logit = np.log(np.clip(opacity, 1e-6, 1-1e-6) / (1 - np.clip(opacity, 1e-6, 1-1e-6)))
    sc_log = np.log(np.clip(scaling, 1e-6, None))

    # PLY data structure
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
             ('opacity', 'f4'),
             ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
             ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')]
    
    N = xyz.shape[0]
    elements = np.empty(N, dtype=dtype)
    elements['x'], elements['y'], elements['z'] = xyz[:,0], xyz[:,1], xyz[:,2]
    elements['f_dc_0'], elements['f_dc_1'], elements['f_dc_2'] = features[:,0], features[:,1], features[:,2]
    elements['opacity'] = op_logit.flatten()
    elements['scale_0'], elements['scale_1'], elements['scale_2'] = sc_log[:,0], sc_log[:,1], sc_log[:,2]
    elements['rot_0'], elements['rot_1'], elements['rot_2'], elements['rot_3'] = rotation[:,0], rotation[:,1], rotation[:,2], rotation[:,3]
    
    # Normals are zero-filled
    elements['nx'] = elements['ny'] = elements['nz'] = 0

    PlyData([PlyElement.describe(elements, 'vertex')]).write(filename)
    print(f"Saved: {filename}")

# --- Main execution ---
if __name__ == "__main__":
    # Create fireworks model
    model = FireworksSpacetimeModel(num_gaussians=4000)
    
    # Generate 6-stage animation frames
    time_steps = [0.0, 0.3, 0.7, 1.0, 1.3, 1.8]
    descriptions = [
        "Launch begins",
        "During ascent", 
        "Immediately after explosion",
        "Maximum expansion",
        "Fade begins",
        "Fading away"
    ]
    
    for i, (t, desc) in enumerate(zip(time_steps, descriptions)):
        data = model.get_splat_at_t(t)
        filename = f"fireworks_stage{i+1}_t{t:.1f}.ply"
        save_ply(data, filename)
        print(f"Stage {i+1}: {desc} (t={t:.1f})")
    
    print("\n🎆 Fireworks 6-stage animation generation complete!")
    print("Check the generated PLY files in Super Splat Viewer or similar.")
    print("Time progression:")
    print("1. Launch → 2. Ascent → 3. Explosion → 4. Expansion → 5. Fade → 6. Disappearance")
