import gradio as gr
import cv2
import torch
import numpy as np
import torch
import torch.nn.functional as F
from color_transfer import ColorTransfer

def kmeans_gpu_soft(x, n_clusters=20, n_iter=10, temperature=1.0):
    """
    temperature: more soft when larger, more hard when smaller
    """
    N, C = x.shape
    device = x.device

    # initialize centers randomly
    indices = torch.randperm(N, device=device)[:n_clusters]
    centers = x[indices]

    for _ in range(n_iter):
        # calculate distances and get soft assignments via softmax
        dist = torch.cdist(x, centers)  # [N, K]
        weights = F.softmax(-dist / temperature, dim=1)  # [N, K]
        
        # assign new centers based on soft assignments
        new_centers = torch.mm(weights.T, x) / (weights.sum(dim=0, keepdim=True).T + 1e-8)
        centers = new_centers
    
    return weights, centers

def bilateral_smooth(img, d=9, sigma_color=75, sigma_space=75):
    # blateral filter smoothing
    img_np = img.cpu().numpy().astype(np.uint8)
    smoothed = cv2.bilateralFilter(img_np, d, sigma_color, sigma_space)
    
    del img_np
    torch.cuda.empty_cache()
    return torch.from_numpy(smoothed).float().to(img.device)

def sharpen(img, sigma=1.0, strength=1.0):
    # Unsharp mask sharpening using Gaussian blur
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
    
    # Return to [H, W, C] format
    sharpened = sharpened.squeeze(0).permute(1, 2, 0)
    
    del img, blurred, kernel_h, kernel_v, gaussian_kernel
    torch.cuda.empty_cache()
    return sharpened.clamp(0, 255)

# flat color quantization with multi-scale processing
def flat_color_multi_scale(image_input, n_colors=20, scales=[1.0, 0.5, 0.25], 
                          spatial_scale=50, temperature=2.0, sharpen_strength=1.0):
    """
    Multi-scale flat color processing    
    """
    with torch.no_grad():
        img_rgb = image_input
        if img_rgb.shape[2] != 3:
            raise ValueError("Input image must have 3 channels (RGB).")

        h, w, _ = img_rgb.shape

        results = []
        
        for scale in scales:
            # scale image
            new_h, new_w = int(h * scale), int(w * scale)
            img_resized = cv2.resize(img_rgb, (new_w, new_h))
            
            # convert to Tensor
            img_tensor = torch.from_numpy(img_resized).float().to("cuda")
            img_flat = img_tensor.reshape(-1, 3)
            
            # spatial features
            xx, yy = torch.meshgrid(torch.arange(new_h, device="cuda"),
                                    torch.arange(new_w, device="cuda"),
                                    indexing="ij")
            features = torch.cat([
                img_flat,
                xx.reshape(-1, 1).float() / new_h * spatial_scale * scale,
                yy.reshape(-1, 1).float() / new_w * spatial_scale * scale
            ], dim=1)
            
            # clustering
            weights, centers = kmeans_gpu_soft(features, n_clusters=n_colors, 
                                            temperature=temperature)
            
            # reconstruction
            rgb_centers = centers[:, :3]
            dist_to_pixels = torch.cdist(rgb_centers, img_flat)
            closest_pixel_indices = dist_to_pixels.argmin(dim=1)
            new_colors = img_flat[closest_pixel_indices]
            
            out_flat = torch.mm(weights, new_colors)
            out = out_flat.reshape(new_h, new_w, 3)
            
            # resize back to original size
            out_np = out.clamp(0, 255).byte().cpu().numpy()
            out_resized = cv2.resize(out_np, (w, h))
            results.append(torch.from_numpy(out_resized).float().to("cuda"))
            
            del img_resized, img_flat, img_tensor, features, weights, centers, out_flat, out, out_np, out_resized, rgb_centers, dist_to_pixels, closest_pixel_indices, new_colors
            torch.cuda.empty_cache()
        
        # fuse multi-scale results
        final_result = sum(results) / len(results)
        
        # post-processing - bilateral smoothing
        final_result = bilateral_smooth(final_result, d=11, sigma_color=35, sigma_space=70)
        
        # post-processing - sharpening
        if sharpen_strength > 0:
            final_result = sharpen(final_result, sigma=1.0, strength=sharpen_strength)
        
        out_img = final_result.clamp(0, 255).byte().cpu().numpy()
        
        # Clean up final tensors and clear VRAM
        del final_result, results
        torch.cuda.empty_cache()
            
        return out_img

def process_image(input_image, n_colors, temperature, spatial_scale, sharpen_strength, color_transfer):
    if input_image is None:
        return None

    processed_img = flat_color_multi_scale(input_image, 
                                         n_colors=n_colors,
                                         temperature=temperature,
                                         spatial_scale=spatial_scale,
                                         sharpen_strength=sharpen_strength)
    
    torch.cuda.empty_cache()
    
    if color_transfer != "None":
        ct = ColorTransfer()
        if color_transfer == "Mean":
            processed_img = ct.mean_std_transfer(img_arr_ref=input_image, img_arr_in=processed_img)
        elif color_transfer == "Lab":
            processed_img = ct.lab_transfer(img_arr_ref=input_image, img_arr_in=processed_img)
        elif color_transfer == "Pdf":
            processed_img = ct.pdf_transfer(img_arr_ref=input_image, img_arr_in=processed_img, regrain=False)
        elif color_transfer == "Pdf+Regrain":
            processed_img = ct.pdf_transfer(img_arr_ref=input_image, img_arr_in=processed_img, regrain=True)
    
    return processed_img

with gr.Blocks() as demo:
    gr.Markdown("# Flat Color multi-scale image processing")
    with gr.Row():
        with gr.Column():
            process_button = gr.Button("Start Processing", variant="primary")
            color_transfer = gr.Dropdown(choices=["Mean", "Lab", "Pdf", "Pdf+Regrain", "None"], value="None", label="Image Color Transfer")
            sharpen_strength = gr.Slider(minimum=0, maximum=10, value=0, step=0.1, label="Sharpen Strength 0=off")
        with gr.Column():
            n_colors = gr.Slider(minimum=2, maximum=1024, value=512, step=1, label="Color Count, more colors = more VRAM")
            temperature = gr.Slider(minimum=1, maximum=20, value=9, step=0.1, label="Temperature")
            spatial_scale = gr.Slider(minimum=1, maximum=500, value=160, step=1, label="Spatial Scale")            
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(label="upload image", type="numpy")
        with gr.Column():
            output_img = gr.Image(format="png", label="Output Image")
    
    process_button.click(
        fn=process_image,
        inputs=[input_img, n_colors, temperature, spatial_scale, sharpen_strength, color_transfer],
        outputs=output_img
    )

if __name__ == "__main__":
    demo.launch()