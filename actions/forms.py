from django import forms

class ImageUploadForm(forms.Form):
    image = forms.ImageField(label='Select an image')
    filter_choice = forms.ChoiceField(
        label='Filter Type',
        choices=[
            ('gaussian', 'Gaussian Blur'),
            ('box', 'Box Blur (Mean)'),
            ('median', 'Median Filter'),
            ('sharpen', 'Sharpen'),
            ('edge', 'Edge Detection'),
            ('histogram_eq', 'Histogram Equalization'),
            ('custom', 'Custom Convolution'),
        ]
    )
    kernel_size = forms.ChoiceField(
        label='Kernel Size',
        choices=[('3', '3x3'), ('5', '5x5'), ('7', '7x7')],
        initial='3'
    )
    custom_kernel = forms.CharField(
        label='Custom Kernel (comma-separated, row-major)',
        required=False,
        help_text='Example for 3x3: 1,1,1,1,1,1,1,1,1'
    )
    continue_filtering = forms.BooleanField(
        label='Continue filtering with current result',
        required=False
    )