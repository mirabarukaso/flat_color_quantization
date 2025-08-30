import os
import cv2
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
import numpy as np
from spandrel import ModelLoader

def kmeans_gpu_soft(x, n_clusters=20, n_iter=10, temperature=1.0):
    """
    Soft clustering version of K-means, using a temperature parameter to control clustering hardness
    temperature: Lower values result in harder clustering, higher values result in softer clustering
    """
    with torch.no_grad():
        N, _ = x.shape
        device = x.device

        # Randomly initialize cluster centers
        indices = torch.randperm(N, device=device)[:n_clusters]
        centers = x[indices]

        for _ in range(n_iter):
            # Calculate distances and use softmax to obtain soft assignments
            dist = torch.cdist(x, centers)  # [N, K]
            weights = F.softmax(-dist / temperature, dim=1)  # [N, K]
            
            # Recalculate centers based on soft assignments
            new_centers = torch.mm(weights.T, x) / (weights.sum(dim=0, keepdim=True).T + 1e-8)
            centers = new_centers
            del dist  # Explicitly free memory
        
        return weights, centers

def sharpen(img, sigma=1.0, strength=1.0):
    """Unsharp mask sharpening using Gaussian blur"""
    with torch.no_grad():
        # Ensure image is in [C, H, W] format for convolution
        img = img.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
        
        # Create Gaussian kernel
        kernel_size = int(6 * sigma + 1)  # Ensure odd kernel size
        if kernel_size % 2 == 0:
            kernel_size += 1
        gaussian_kernel = torch.exp(-torch.arange(-(kernel_size//2), kernel_size//2 + 1, device=img.device)**2 / (2 * sigma**2))
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
        
        # Reshape for horizontal and vertical kernels
        kernel_h = gaussian_kernel.view(1, 1, 1, kernel_size).repeat(3, 1, 1, 1)  # [3, 1, 1, kernel_size]
        kernel_v = gaussian_kernel.view(1, 1, kernel_size, 1).repeat(3, 1, 1, 1)  # [3, 1, kernel_size, 1]
        
        # Apply separable Gaussian blur
        blurred = F.conv2d(img, kernel_h, groups=3, padding=(0, kernel_size//2))
        blurred = F.conv2d(blurred, kernel_v, groups=3, padding=(kernel_size//2, 0))
        
        # Compute unsharp mask
        sharpened = img + strength * (img - blurred)
        
        del blurred, kernel_h, kernel_v  # Free memory
        
        # Return to [H, W, C] format
        sharpened = sharpened.squeeze(0).permute(1, 2, 0)
        return sharpened.clamp(0, 255)

def process_block(img_block, n_colors, spatial_scale, scale, temperature, device="cuda"):
    """Process a single image block with clustering and reconstruction"""
    h, w, _ = img_block.shape
    img_tensor = torch.from_numpy(img_block).float().to(device)
    img_flat = img_tensor.reshape(-1, 3)
    
    # Spatial features
    xx, yy = torch.meshgrid(torch.arange(h, device=device),
                            torch.arange(w, device=device),
                            indexing="ij")
    
    features = torch.cat([
        img_flat,
        xx.reshape(-1, 1).float() / h * spatial_scale * scale,
        yy.reshape(-1, 1).float() / w * spatial_scale * scale
    ], dim=1)
    
    # Clustering
    weights, centers = kmeans_gpu_soft(features, n_clusters=n_colors, temperature=temperature)
    
    # Reconstruction
    rgb_centers = centers[:, :3]
    dist_to_pixels = torch.cdist(rgb_centers, img_flat)
    closest_pixel_indices = dist_to_pixels.argmin(dim=1)
    new_colors = img_flat[closest_pixel_indices]
    
    out_flat = torch.mm(weights, new_colors)
    out = out_flat.reshape(h, w, 3)
    
    del img_flat, xx, yy, features, weights, centers, dist_to_pixels, closest_pixel_indices, out_flat  # Free memory
    return out

class FlatColorizer:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.upscale_model = None
        self.model_scale = None

    # =========================================================
    # 1. Load ESRGAN / RealESRGAN model using spandrel
    # =========================================================
    def _load_upscale_model(self, model_path: str):
        if not os.path.exists(model_path):
            print(f"[WARN] Model file does not exist: {model_path}")
            return False

        try:
            print(f"[INFO] Loading upsampling model: {model_path}")

            # 1. Load original checkpoint
            if model_path.endswith(".safetensors"):
                state_dict = load_file(model_path)
            elif model_path.endswith(".pth"):
                state_dict = torch.load(model_path, map_location=self.device)
            else:
                raise ValueError("Only .safetensors or .pth models are supported!")

            # 2. Parse the actual weights
            if isinstance(state_dict, dict):
                if "params_ema" in state_dict:
                    state_dict = state_dict["params_ema"]
                    print("[INFO] Using params_ema weights")
                elif "params" in state_dict:
                    state_dict = state_dict["params"]
                    print("[INFO] Using params weights")
            else:
                raise RuntimeError("Invalid model file: Weight format is incompatible")

            # 3. Load model using spandrel
            model = ModelLoader().load_from_state_dict(state_dict).eval().to(self.device).half()
            self.model_scale = model.scale
            self.upscale_model = model

            print(f"[INFO] Successfully loaded model: {type(model).__name__} | Built-in scale: {self.model_scale}x")
            return True

        except Exception as e:
            print(f"[ERROR] Failed to load model {model_path}: {e}")
            self.upscale_model = None
            self.model_scale = None
            return False

    # =========================================================
    # 2. Neural network super-sampling
    # =========================================================
    def _nn_super_sample(self, img_rgb: np.ndarray, target_scale: float):
        """Neural network super-sampling with proper tensor handling"""
        if self.upscale_model is None:
            return None
        
        try:
            h, w, c = img_rgb.shape
            print(f"[DEBUG] Input image dimensions: {h}x{w}, target scale: {target_scale}")
            
            # Determine processing strategy based on image size
            max_size = 1024  # Adjust threshold to avoid memory overflow
            if h > max_size or w > max_size:
                print(f"[INFO] Image is too large ({h}x{w}), using tiled processing")
                return self._tile_process_image(img_rgb, target_scale)
            
            # Prepare input tensor
            img_t = torch.from_numpy(img_rgb.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
            img_t = img_t / 255.0  # Normalize to [0, 1]
            
            # Move to device and convert precision
            img_t = img_t.to(self.device)
            if self.device == "cuda" and torch.cuda.is_available():
                img_t = img_t.half()

            print(f"[DEBUG] Input tensor shape: {img_t.shape}, device: {img_t.device}, dtype: {img_t.dtype}")

            # Neural network inference
            with torch.no_grad():
                torch.cuda.empty_cache()  # Clear cache
                out_t = self.upscale_model(img_t)
                
            print(f"[DEBUG] Model output tensor shape: {out_t.shape}")

            # Clamp to valid range
            out_t = torch.clamp(out_t, 0, 1)

            # Handle scale mismatch
            actual_scale = out_t.shape[-1] / img_t.shape[-1]  # Calculate actual scale
            print(f"[DEBUG] Model actual scale: {actual_scale}, expected scale: {target_scale}")
            
            if abs(actual_scale - target_scale) > 0.1:
                scale_ratio = target_scale / actual_scale
                print(f"[DEBUG] Secondary scaling required, ratio: {scale_ratio}")
                out_t = F.interpolate(
                    out_t, scale_factor=scale_ratio,
                    mode="bicubic", align_corners=False
                )
                print(f"[DEBUG] Shape after secondary scaling: {out_t.shape}")

            # Convert back to numpy
            out_img = out_t.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
            out_img = (out_img * 255.0).astype(np.uint8)
            
            # Clear GPU memory
            del img_t, out_t
            torch.cuda.empty_cache()
            
            print(f"[DEBUG] Final output image dimensions: {out_img.shape}")
            return out_img
            
        except Exception as e:
            print(f"[ERROR] Neural network super-sampling failed, falling back to OpenCV: {e}")
            import traceback
            traceback.print_exc()
            # Clear possible GPU memory usage
            if 'img_t' in locals():
                del img_t
            if 'out_t' in locals():
                del out_t
            torch.cuda.empty_cache()
            return None
        
    def _tile_process_image(self, img_rgb: np.ndarray, target_scale: float, tile_size=512, overlap=32):
        """Process large images in tiles to avoid memory issues"""
        h, w, _ = img_rgb.shape
        scale = self.model_scale or 4
        
        # Calculate output dimensions
        out_h, out_w = int(h * scale), int(w * scale)
        output = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        
        print(f"[DEBUG] Tiled processing: Input {h}x{w}, expected output {out_h}x{out_w}, tile size {tile_size}")
        
        tiles_processed = 0
        total_tiles = ((h - 1) // (tile_size - overlap) + 1) * ((w - 1) // (tile_size - overlap) + 1)
        
        # Process each tile
        for i in range(0, h, tile_size - overlap):
            for j in range(0, w, tile_size - overlap):
                tiles_processed += 1
                print(f"[DEBUG] Processing tile {tiles_processed}/{total_tiles}")
                
                # Extract tile
                i_end = min(i + tile_size, h)
                j_end = min(j + tile_size, w)
                tile = img_rgb[i:i_end, j:j_end]
                
                # Process single tile
                tile_out = self._process_single_tile(tile)
                if tile_out is None:
                    continue
                
                # Calculate output position
                out_i = int(i * scale)
                out_j = int(j * scale)
                out_i_end = min(int(i_end * scale), out_h)
                out_j_end = min(int(j_end * scale), out_w)
                
                # Adjust tile output size to match target area
                tile_h, tile_w = out_i_end - out_i, out_j_end - out_j
                if tile_out.shape[:2] != (tile_h, tile_w):
                    tile_out = cv2.resize(tile_out, (tile_w, tile_h))
                
                # Place into output
                output[out_i:out_i_end, out_j:out_j_end] = tile_out
                
                # Clear GPU memory
                del tile_out  # Free memory
                torch.cuda.empty_cache()
        
        # Handle final scale adjustment
        if abs(scale - target_scale) > 0.1:
            final_h, final_w = int(h * target_scale), int(w * target_scale)
            output = cv2.resize(output, (final_w, final_h))
        
        return output
    
    def _process_single_tile(self, tile: np.ndarray):
        """Process a single tile through the model"""
        try:
            # Prepare tensor
            tile_t = torch.from_numpy(tile.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
            tile_t = tile_t / 255.0
            tile_t = tile_t.to(self.device)
            
            if self.device == "cuda" and torch.cuda.is_available():
                tile_t = tile_t.half()
            
            # Inference
            with torch.no_grad():
                out_t = self.upscale_model(tile_t)
                
            out_t = torch.clamp(out_t, 0, 1)
            
            # Convert back to numpy
            out_tile = out_t.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
            out_tile = (out_tile * 255.0).astype(np.uint8)
            
            # Clear
            del tile_t, out_t
            
            return out_tile
            
        except Exception as e:
            print(f"[ERROR] Failed to process tile: {e}")
            return None

    # =========================================================
    # 3. OpenCV super-sampling
    # =========================================================
    def _cv2_super_sample(self, img_rgb: np.ndarray, scale: float):
        h, w = img_rgb.shape[:2]
        if scale <= 1.0:
            return img_rgb
        return cv2.resize(
            img_rgb, (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_CUBIC
        )

    # =========================================================
    # 4. Unified super-sampling entry
    # =========================================================
    def super_sample(self, img_rgb: np.ndarray, scale: float, model_path: str = None):
        if model_path and self._load_upscale_model(model_path):
            out = self._nn_super_sample(img_rgb, scale)
            if out is not None:
                return out
        # Model loading failed / inference failed -> Fall back to OpenCV
        return self._cv2_super_sample(img_rgb, scale)
    
    # =========================================================
    # 5. Process GPU soft
    # =========================================================
    def start_process(self, img_rgb, n_colors, spatial_scale, scales, temperature, block_size):
        h, w, _ = img_rgb.shape
        results = []

        print(f"[INFO] Starting multi-scale processing with scales: {scales}")
        print(f"[DEBUG] Input image dimensions: {h}x{w}")

        for scale in scales:
            # Resize image
            new_h, new_w = int(h * scale), int(w * scale)
            print(f"[DEBUG] Processing scale {scale}, resizing to {new_h}x{new_w}")
            img_resized = cv2.resize(img_rgb, (new_w, new_h))
            
            # Check if block processing is needed
            if new_h > block_size or new_w > block_size:
                print(f"[INFO] Image size {new_h}x{new_w} exceeds block size {block_size}, using block processing")
                block_results = np.zeros((new_h, new_w, 3), dtype=np.uint8)
                step_h = block_size
                step_w = block_size
                
                total_blocks = ((new_h - 1) // step_h + 1) * ((new_w - 1) // step_w + 1)
                block_count = 0
                
                for i in range(0, new_h, step_h):
                    for j in range(0, new_w, step_w):
                        block_count += 1
                        # Extract block, slightly expand boundaries to avoid seams
                        i_end = min(i + step_h, new_h)
                        j_end = min(j + step_w, new_w)
                        block = img_resized[i:i_end, j:j_end]
                        
                        print(f"[DEBUG] Processing block {block_count}/{total_blocks}, position ({i},{j}) to ({i_end},{j_end}), size {block.shape[:2]}")
                        
                        # Process block
                        out_block = process_block(block, n_colors, spatial_scale, scale, temperature)
                        out_block_np = out_block.clamp(0, 255).byte().cpu().numpy()
                        block_results[i:i_end, j:j_end] = out_block_np
                                                
                        # Clear GPU memory
                        torch.cuda.empty_cache()
                
                print(f"[DEBUG] Resizing block results to original dimensions: {w}x{h}")
                out_resized = cv2.resize(block_results, (w, h))
                results.append(torch.from_numpy(out_resized).float())  # Keep on CPU
                print(f"[DEBUG] Appended block-processed result, shape: {out_resized.shape}")
            else:
                print(f"[INFO] Image size {new_h}x{new_w} within block size, processing directly")
                # Directly process the entire resized image
                out = process_block(img_resized, n_colors, spatial_scale, scale, temperature)
                out_np = out.clamp(0, 255).byte().cpu().numpy()
                print(f"[DEBUG] Direct processing completed, output shape: {out_np.shape}")
                out_resized = cv2.resize(out_np, (w, h))
                results.append(torch.from_numpy(out_resized).float())  # Keep on CPU
                print(f"[DEBUG] Appended direct-processed result, shape: {out_resized.shape}")
        
        # Combine multi-scale results
        final_result = sum(results) / len(results)
        final_result = final_result.to("cuda") if self.device == "cuda" else final_result
        return final_result
    
    # =========================================================
    # 6. Main processing 
    # =========================================================
    def unload_model(self):
        if self.upscale_model is not None:
            self.upscale_model = None
            self.model_scale = None
            torch.cuda.empty_cache()
            print("[INFO] Unloaded upscale model from memory")
            
    def flat_color_multi_scale(
        self,
        img_path,
        n_colors=512,
        temperature=2,
        spatial_scale=80,
        sharpen_strength=0.0,
        block_size=512,
        upscale=1.0,
        model_path=None,
        denoising=None,
        img_rgb=None
    ):
        if img_path != None:
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                raise FileNotFoundError(f"Failed to load image: {img_path}")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
        original_h, original_w = img_rgb.shape[:2]

        # Step 1. Super-sampling        
        if upscale > 1.001:
            if upscale > 8:
                upscale = 8
                print(f"[INFO] Upscale too big, reset to: {upscale}")
            # Apply a light denoising step before super-sampling to ensure the input to the neural model is clean.
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            img_smoothed = cv2.GaussianBlur(img_bgr, (3, 3), 1, cv2.BORDER_REPLICATE) # sigma 1.0
            img_rgb = cv2.cvtColor(img_smoothed, cv2.COLOR_BGR2RGB)
            img_rgb = self.super_sample(img_rgb, upscale, model_path)            

        # Step 2. Call original flat_color's process_block
        scales = [1.0, 0.75, 0.5, 0.25]
        final_result = self.start_process(img_rgb, n_colors, spatial_scale, scales, temperature, block_size)
            
        # Step 3. Denoising
        if denoising:
            print("[INFO] Applying denoising after multi-scale processing")
            final_result_np = final_result.clamp(0, 255).byte().cpu().numpy()
            final_result_bgr = cv2.cvtColor(final_result_np, cv2.COLOR_RGB2BGR)
            
            h_d, h_color, template_indow_size, search_window_size = denoising            
            print(f"[DEBUG] Denoising parameters: h={h_d}, h_color={h_color}, template_indow_size={template_indow_size}, search_window_size={search_window_size}")
            
            final_result_bgr = cv2.fastNlMeansDenoisingColored(
                    final_result_bgr,
                    h=h_d,
                    hColor=h_color,
                    templateWindowSize=template_indow_size,
                    searchWindowSize=search_window_size
                )                
            
            final_result = torch.from_numpy(cv2.cvtColor(final_result_bgr, cv2.COLOR_BGR2RGB)).float().to("cuda")        
        
        # Step 4. Post-processing - Sharpening    
        if sharpen_strength > 0.001:
            print(f"[INFO] Applying sharpening with strength: {sharpen_strength}")
            final_result = sharpen(final_result, sigma=1.0, strength=sharpen_strength)

        # Step 5: Resize back to original dimensions if upscaled
        if upscale > 1.001:                
                current_h, current_w = final_result.shape[:2]
                
                if current_h != original_h or current_w != original_w:
                    print(f"[INFO] Resizing result from {current_h}x{current_w} back to original dimensions {original_h}x{original_w}")
                    final_result_float = final_result.float() / 255.0
                    final_result_float = final_result_float.permute(2, 0, 1).unsqueeze(0)
                    final_result_resized = F.interpolate(
                        final_result_float,
                        size=(original_h, original_w),
                        mode='bicubic',
                        align_corners=False,
                        antialias=True
                    )
                    final_result_resized = final_result_resized.squeeze(0).permute(1, 2, 0)
                    final_result = (final_result_resized * 255.0).clamp(0, 255)
                
                print(f"[DEBUG] Resizing completed: shape {final_result.shape}")
        
        out_img = final_result.clamp(0, 255).byte().cpu().numpy()
        torch.cuda.empty_cache()                
        return out_img

# =========================================================
# CLI entry
# =========================================================
if __name__ == "__main__":
    import argparse
    import time

    start_time = time.time()
    
    def parse_ints(s):
        parts = [int(v) for v in s.split(',')]
        if len(parts) != 4:
            print(f"[ERROR] Argument error {s}")
            print("[INFO] Use default [4,4,5,15]")
            return [4,4,5,15]              
        return parts   
    
    parser = argparse.ArgumentParser(description="Flat Colorizer with Super-sampling & ESRGAN/RealESRGAN")
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--output", required=True, help="Output image path")
    parser.add_argument("--n_colors", type=int, default=256, help="Number of High-colors (256-4096)")
    parser.add_argument("--temperature", type=float, default=2.0, help="Soft assignment temperature (<5.0)")
    parser.add_argument("--spatial_scale", type=int, default=60, help="Spatial coordinate weight (40-500)")
    parser.add_argument("--sharpen_strength", type=float, default=0.0, help="Sharpening strength (0=disabled)")
    parser.add_argument("--block_size", type=int, default=512, help="Block size")
    parser.add_argument("--upscale", type=float, default=1.0, help="Super-sampling scale (1.0=disabled, max 3.0)")
    parser.add_argument("--upscale_model", type=str, default=None, help="Upscale (NMKD/ESRGAN/RealESRGAN) model path (.safetensors/.pth)")
    parser.add_argument("--denoise", type=parse_ints, help="Enable post-denoising [4,4,5,15]/[5,7,5,21]")
    args = parser.parse_args()
    
    fc = FlatColorizer()
    out = fc.flat_color_multi_scale(
        args.input,
        n_colors=args.n_colors,
        temperature=args.temperature,
        spatial_scale=args.spatial_scale,
        sharpen_strength=args.sharpen_strength,
        block_size=args.block_size,
        upscale=args.upscale,
        model_path=args.upscale_model,
        denoising=args.denoise
    )
    cv2.imwrite(args.output, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    print(f"[INFO] Output completed: {args.output}")
    
    end_time = time.time()
    print(f"[INFO] Total time: {end_time - start_time:.2f} seconds")