from django.shortcuts import render
# filters_app/views.py
import os
from django.shortcuts import render, redirect
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse
from .forms import ImageUploadForm

# filters_app/views.py (updated apply_filter function)
import cv2  # Add this import at the top
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from scipy import ndimage  #  for better filtering


def apply_filter(image, filter_type, kernel_size=3, custom_kernel=None):
    """Apply filter to image with improved edge detection"""
    kernel_size = int(kernel_size)

    # Convert PIL Image to numpy array
    img_array = np.array(image)

    if filter_type == 'edge_laplacian':
        # Improved Laplacian edge detection
        if img_array.ndim == 3:  # Color image
            # Convert to grayscale first
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        # Apply Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)

        # Normalize to 0-255
        laplacian = np.abs(laplacian)
        laplacian = np.clip(laplacian, 0, 255)
        laplacian = laplacian.astype(np.uint8)

        return Image.fromarray(laplacian)

    elif filter_type == 'edge_sobel':
        # Sobel edge detection (better for most cases)
        if img_array.ndim == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        # Sobel in X and Y directions
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)

        # Calculate gradient magnitude
        gradient_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)

        # Normalize and scale
        gradient_magnitude = np.clip(gradient_magnitude, 0, 255)
        gradient_magnitude = (gradient_magnitude * 255 / gradient_magnitude.max()).astype(np.uint8)

        return Image.fromarray(gradient_magnitude)

    elif filter_type == 'edge_canny':
        # Canny edge detection (most robust)
        if img_array.ndim == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        # Apply Gaussian blur first (reduces noise)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # Canny edge detection with automatic thresholding
        v = np.median(blurred)
        lower = int(max(0, (1.0 - 0.33) * v))
        upper = int(min(255, (1.0 + 0.33) * v))
        edges = cv2.Canny(blurred, lower, upper)

        return Image.fromarray(edges)

    elif filter_type == 'edge_prewitt':
        # Prewitt edge detection
        if img_array.ndim == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        # Prewitt kernels
        kernel_x = np.array([[-1, 0, 1],
                             [-1, 0, 1],
                             [-1, 0, 1]])
        kernel_y = np.array([[-1, -1, -1],
                             [0, 0, 0],
                             [1, 1, 1]])

        # Convolve
        prewitt_x = ndimage.convolve(gray.astype(float), kernel_x)
        prewitt_y = ndimage.convolve(gray.astype(float), kernel_y)

        # Magnitude
        magnitude = np.sqrt(prewitt_x ** 2 + prewitt_y ** 2)

        # Normalize
        magnitude = np.clip(magnitude, 0, 255)
        magnitude = (magnitude * 255 / magnitude.max()).astype(np.uint8)

        return Image.fromarray(magnitude)

    elif filter_type == 'edge_log':
        # Laplacian of Gaussian (LoG) - Marr-Hildreth
        if img_array.ndim == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

        # Apply Laplacian
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F)

        # Find zero crossings
        laplacian = np.abs(laplacian)
        laplacian = np.clip(laplacian, 0, 255)
        laplacian = (laplacian * 255 / laplacian.max()).astype(np.uint8)

        return Image.fromarray(laplacian)

    elif filter_type == 'edge_robinson':
        # Robinson compass edge detection
        if img_array.ndim == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        # Robinson compass masks
        masks = [
            np.array([[-1, 0, 1],
                      [-2, 0, 2],
                      [-1, 0, 1]]),  # North
            np.array([[0, 1, 2],
                      [-1, 0, 1],
                      [-2, -1, 0]]),  # North-East
            np.array([[1, 2, 1],
                      [0, 0, 0],
                      [-1, -2, -1]]),  # East
            np.array([[2, 1, 0],
                      [1, 0, -1],
                      [0, -1, -2]])  # South-East
        ]

        # Apply all masks and take maximum response
        responses = []
        for mask in masks:
            response = ndimage.convolve(gray.astype(float), mask)
            responses.append(np.abs(response))

        # Combine responses
        combined = np.max(responses, axis=0)

        # Normalize
        combined = np.clip(combined, 0, 255)
        combined = (combined * 255 / combined.max()).astype(np.uint8)

        return Image.fromarray(combined)

    # Keep your original filters (gaussian, box, etc.)
    elif filter_type == 'gaussian':
        return image.filter(ImageFilter.GaussianBlur(radius=kernel_size / 2))

    elif filter_type == 'box':
        return image.filter(ImageFilter.BoxBlur(radius=kernel_size / 2))

    elif filter_type == 'median':
        return image.filter(ImageFilter.MedianFilter(size=kernel_size))

    elif filter_type == 'sharpen':
        return image.filter(ImageFilter.SHARPEN)

    elif filter_type == 'histogram_eq':
        if image.mode != 'L':
            image = image.convert('L')
        return ImageOps.equalize(image)

    elif filter_type == 'custom' and custom_kernel:
        try:
            kernel_values = list(map(float, custom_kernel.split(',')))
            n = len(kernel_values)
            size = int(np.sqrt(n))
            if size ** 2 == n:
                # Normalize kernel to avoid overflow
                kernel_sum = sum(kernel_values)
                if kernel_sum != 0:
                    kernel_values = [k / kernel_sum for k in kernel_values]
                return image.filter(ImageFilter.Kernel((size, size), kernel_values))
        except:
            pass

    return image
#
# def apply_filter(image, filter_type, kernel_size=3, custom_kernel=None):
#     """Apply filter to image"""
#     img_array = np.array(image)
#
#     if len(img_array.shape) == 3:  # Color image
#         height, width, channels = img_array.shape
#         is_grayscale = False
#     else:  # Grayscale image
#         height, width = img_array.shape
#         channels = 1
#         is_grayscale = True
#
#     kernel_size = int(kernel_size)
#
#     if filter_type == 'gaussian':
#         # Apply Gaussian blur using PIL
#         return image.filter(ImageFilter.GaussianBlur(radius=kernel_size / 2))
#
#     elif filter_type == 'box':
#         # Box blur (mean filter)
#         return image.filter(ImageFilter.BoxBlur(radius=kernel_size / 2))
#
#     elif filter_type == 'median':
#         # Median filter
#         return image.filter(ImageFilter.MedianFilter(size=kernel_size))
#
#     elif filter_type == 'sharpen':
#         # Sharpen filter
#         kernel = np.array([[-1, -1, -1],
#                            [-1, 9, -1],
#                            [-1, -1, -1]])
#         if kernel_size > 3:
#             kernel = np.ones((kernel_size, kernel_size)) * -1
#             center = kernel_size // 2
#             kernel[center, center] = kernel_size ** 2 + 1
#         return image.filter(ImageFilter.Kernel((kernel_size, kernel_size), kernel.flatten()))
#
#     elif filter_type == 'edge':
#         # Edge detection (Laplacian)
#         kernel = np.array([[-1, -1, -1],
#                            [-1, 8, -1],
#                            [-1, -1, -1]])
#         if kernel_size > 3:
#             kernel = np.ones((kernel_size, kernel_size)) * -1
#             center = kernel_size // 2
#             kernel[center, center] = kernel_size ** 2 - 1
#         return image.filter(ImageFilter.Kernel((kernel_size, kernel_size), kernel.flatten()))
#
#     elif filter_type == 'histogram_eq':
#         # Convert to grayscale if needed
#         if image.mode != 'L':
#             image = image.convert('L')
#         # Apply histogram equalization
#         return ImageOps.equalize(image)
#
#     elif filter_type == 'custom' and custom_kernel:
#         # Parse custom kernel
#         try:
#             kernel_values = list(map(float, custom_kernel.split(',')))
#             n = len(kernel_values)
#             size = int(np.sqrt(n))
#             if size ** 2 == n:
#                 return image.filter(ImageFilter.Kernel((size, size), kernel_values))
#         except:
#             pass
#
#     return image


def upload_image(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)

        if form.is_valid():
            # image upload
            image_file = request.FILES['image']
            fs = FileSystemStorage()

            # original image
            original_filename = fs.save(f'original_{image_file.name}', image_file)
            original_path = fs.path(original_filename)

            # if we're continuing with previous result
            if form.cleaned_data.get('continue_filtering') and 'processed_image' in request.session:
                # Load previous result
                processed_path = request.session['processed_image']
                image = Image.open(processed_path)
            else:
                #  new image   load
                image = Image.open(original_path)

            # Apply selected filter
            filter_type = form.cleaned_data['filter_choice']
            kernel_size = form.cleaned_data['kernel_size']
            custom_kernel = form.cleaned_data.get('custom_kernel', '')

            processed_image = apply_filter(
                image,
                filter_type,
                kernel_size,
                custom_kernel
            )

            # Save processed image
            processed_filename = f'processed_{image_file.name}'
            processed_path = os.path.join(settings.MEDIA_ROOT, processed_filename)
            processed_image.save(processed_path)

            # Save to session for consecutive filtering
            request.session['processed_image'] = processed_path
            request.session['original_image'] = original_path

            # Prepare context for result page
            context = {
                'original_image': fs.url(original_filename),
                'processed_image': f'/media/{processed_filename}',
                'filter_name': dict(form.fields['filter_choice'].choices)[filter_type],
                'kernel_size': f"{kernel_size}x{kernel_size}",
                'form': ImageUploadForm(initial={'continue_filtering': True}),
            }

            return render(request, 'filters_app/result.html', context)

    else:
        form = ImageUploadForm()

    return render(request, 'filters_app/upload.html', {'form': form})


def reset_filtering(request):
    """Reset to start with new image"""
    if 'processed_image' in request.session:
        # Clean up processed image file
        try:
            os.remove(request.session['processed_image'])
        except:
            pass
        del request.session['processed_image']

    if 'original_image' in request.session:
        del request.session['original_image']

    return redirect('upload_image')