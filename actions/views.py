from django.shortcuts import render
# filters_app/views.py
import os
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from django.shortcuts import render, redirect
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse
from .forms import ImageUploadForm


def apply_filter(image, filter_type, kernel_size=3, custom_kernel=None):
    """Apply filter to image"""
    img_array = np.array(image)

    if len(img_array.shape) == 3:  # Color image
        height, width, channels = img_array.shape
        is_grayscale = False
    else:  # Grayscale image
        height, width = img_array.shape
        channels = 1
        is_grayscale = True

    kernel_size = int(kernel_size)

    if filter_type == 'gaussian':
        # Apply Gaussian blur using PIL
        return image.filter(ImageFilter.GaussianBlur(radius=kernel_size / 2))

    elif filter_type == 'box':
        # Box blur (mean filter)
        return image.filter(ImageFilter.BoxBlur(radius=kernel_size / 2))

    elif filter_type == 'median':
        # Median filter
        return image.filter(ImageFilter.MedianFilter(size=kernel_size))

    elif filter_type == 'sharpen':
        # Sharpen filter
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
        if kernel_size > 3:
            kernel = np.ones((kernel_size, kernel_size)) * -1
            center = kernel_size // 2
            kernel[center, center] = kernel_size ** 2 + 1
        return image.filter(ImageFilter.Kernel((kernel_size, kernel_size), kernel.flatten()))

    elif filter_type == 'edge':
        # Edge detection (Laplacian)
        kernel = np.array([[-1, -1, -1],
                           [-1, 8, -1],
                           [-1, -1, -1]])
        if kernel_size > 3:
            kernel = np.ones((kernel_size, kernel_size)) * -1
            center = kernel_size // 2
            kernel[center, center] = kernel_size ** 2 - 1
        return image.filter(ImageFilter.Kernel((kernel_size, kernel_size), kernel.flatten()))

    elif filter_type == 'histogram_eq':
        # Convert to grayscale if needed
        if image.mode != 'L':
            image = image.convert('L')
        # Apply histogram equalization
        return ImageOps.equalize(image)

    elif filter_type == 'custom' and custom_kernel:
        # Parse custom kernel
        try:
            kernel_values = list(map(float, custom_kernel.split(',')))
            n = len(kernel_values)
            size = int(np.sqrt(n))
            if size ** 2 == n:
                return image.filter(ImageFilter.Kernel((size, size), kernel_values))
        except:
            pass

    return image


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