import argparse
import torch
import nibabel as nib
import numpy as np
from typing import Tuple, Union, List
import logging
from functools import wraps

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Custom Exceptions
class NIFTIToolboxError(Exception):
    """Base exception for NIFTI Toolbox errors."""
    pass

class FileIOError(NIFTIToolboxError):
    """Raised when there's an error in file I/O operations."""
    pass

class InvalidOperationError(NIFTIToolboxError):
    """Raised when an invalid operation is attempted."""
    pass

class DataTypeError(NIFTIToolboxError):
    """Raised when there's a data type mismatch or invalid data type."""
    pass

class DimensionalityError(NIFTIToolboxError):
    """Raised when there's a mismatch in image dimensions."""
    pass

# Decorator for error handling
def error_handler(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except NIFTIToolboxError as e:
            logger.error(f"{func.__name__} failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
            raise NIFTIToolboxError(f"Unexpected error in {func.__name__}: {str(e)}")
    return wrapper

@error_handler
def load_nifti(file_path: str) -> Tuple[torch.Tensor, np.ndarray, nib.Nifti1Header]:
    """Load a NIFTI file and return a PyTorch tensor on GPU."""
    try:
        img = nib.load(file_path)
        data = img.get_fdata()
        return torch.tensor(data, device='cuda'), img.affine, img.header
    except FileNotFoundError:
        raise FileIOError(f"File not found: {file_path}")
    except nib.filebasedimages.ImageFileError:
        raise FileIOError(f"Invalid NIFTI file: {file_path}")

@error_handler
def save_nifti(data: torch.Tensor, affine: np.ndarray, header: nib.Nifti1Header, output_path: str):
    """Save a PyTorch tensor as a NIFTI file."""
    try:
        data_np = data.cpu().numpy()
        new_img = nib.Nifti1Image(data_np, affine, header)
        nib.save(new_img, output_path)
    except Exception as e:
        raise FileIOError(f"Could not save NIFTI file: {str(e)}")

@error_handler
def add(tensor1: torch.Tensor, tensor2: Union[torch.Tensor, float]) -> torch.Tensor:
    """Add two tensors or a tensor and a scalar."""
    return tensor1 + tensor2

@error_handler
def sub(tensor1: torch.Tensor, tensor2: Union[torch.Tensor, float]) -> torch.Tensor:
    """Subtract tensor2 from tensor1."""
    return tensor1 - tensor2

@error_handler
def mul(tensor1: torch.Tensor, tensor2: Union[torch.Tensor, float]) -> torch.Tensor:
    """Multiply two tensors or a tensor and a scalar."""
    return tensor1 * tensor2

@error_handler
def div(tensor1: torch.Tensor, tensor2: Union[torch.Tensor, float]) -> torch.Tensor:
    """Divide tensor1 by tensor2."""
    return tensor1 / tensor2

@error_handler
def mas(tensor1: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Apply a binary mask to a tensor."""
    if tensor1.shape != mask.shape:
        raise DimensionalityError("Tensor and mask must have the same shape")
    return tensor1 * mask

@error_handler
def thr(tensor: torch.Tensor, threshold: float) -> torch.Tensor:
    """Apply a lower threshold to a tensor."""
    return torch.where(tensor >= threshold, tensor, torch.tensor(0.0, device='cuda'))

@error_handler
def uthr(tensor: torch.Tensor, upper_threshold: float) -> torch.Tensor:
    """Apply an upper threshold to a tensor."""
    return torch.where(tensor <= upper_threshold, tensor, torch.tensor(0.0, device='cuda'))

@error_handler
def binarize(tensor: torch.Tensor) -> torch.Tensor:
    """Binarize the input tensor."""
    return torch.where(tensor > 0, torch.tensor(1.0, device='cuda'), torch.tensor(0.0, device='cuda'))

@error_handler
def binv(tensor: torch.Tensor) -> torch.Tensor:
    """Invert binarization of the input tensor."""
    return torch.where(tensor == 0, torch.tensor(1.0, device='cuda'), torch.tensor(0.0, device='cuda'))

def print_examples():
    examples = """
    Example Commands:

    Basic Operations:
    - Add a scalar or another NIFTI image:
      python gnifti-maths.py -i input.nii -o output.nii --add 5
      python gnifti-maths.py -i input.nii -o output.nii --add another_image.nii

    - Subtract a scalar or another NIFTI image:
      python gnifti-maths.py -i input.nii -o output.nii --sub 3
      python gnifti-maths.py -i input.nii -o output.nii --sub another_image.nii

    - Multiply by a scalar or another NIFTI image:
      python gnifti-maths.py -i input.nii -o output.nii --mul 2
      python gnifti-maths.py -i input.nii -o output.nii --mul another_image.nii

    - Divide by a scalar or another NIFTI image:
      python gnifti-maths.py -i input.nii -o output.nii --div 2
      python gnifti-maths.py -i input.nii -o output.nii --div another_image.nii

    Masking and Thresholding:
    - Apply a binary mask:
      python gnifti-maths.py -i input.nii -o output.nii --mas mask_image.nii

    - Apply a lower threshold:
      python gnifti-maths.py -i input.nii -o output.nii --thr 0.5

    - Apply an upper threshold:
      python gnifti-maths.py -i input.nii -o output.nii --uthr 1.5

    Advanced Operations:
    - Apply the exponential function:
      python gnifti-maths.py -i input.nii -o output.nii --exp

    - Apply the logarithm function:
      python gnifti-maths.py -i input.nii -o output.nii --log

    - Apply the sine function:
      python gnifti-maths.py -i input.nii -o output.nii --sin

    - Apply the cosine function:
      python gnifti-maths.py -i input.nii -o output.nii --cos

    - Apply the tangent function:
      python gnifti-maths.py -i input.nii -o output.nii --tan

    - Apply the inverse sine function:
      python gnifti-maths.py -i input.nii -o output.nii --asin

    - Apply the inverse cosine function:
      python gnifti-maths.py -i input.nii -o output.nii --acos

    - Apply the inverse tangent function:
      python gnifti-maths.py -i input.nii -o output.nii --atan

    - Square the input image:
      python gnifti-maths.py -i input.nii -o output.nii --sqr

    - Apply the square root function:
      python gnifti-maths.py -i input.nii -o output.nii --sqrt

    - Apply the reciprocal function:
      python gnifti-maths.py -i input.nii -o output.nii --recip

    - Apply the absolute value function:
      python gnifti-maths.py -i input.nii -o output.nii --abs

    - Binarize the input image:
      python gnifti-maths.py -i input.nii -o output.nii --binarize

    - Invert the binarization of the input image:
      python gnifti-maths.py -i input.nii -o output.nii --binv

    Dilation and Erosion:
    - Apply mean dilation:
      python gnifti-maths.py -i input.nii -o output.nii --dilM 3 3 3

    - Apply modal dilation:
      python gnifti-maths.py -i input.nii -o output.nii --dilD 3 3 3

    - Apply erosion:
      python gnifti-maths.py -i input.nii -o output.nii --ero 3 3 3

    TFCE and Tensor Decomposition:
    - Apply TFCE (Threshold-Free Cluster Enhancement):
      python gnifti-maths.py -i input.nii -o output.nii --tfce 0.5 2.0

    - Perform tensor decomposition and return Fractional Anisotropy (FA) image:
      python gnifti-maths.py -i input.nii -o output.nii --tensor_decomp

    Bayesian Polynomial Trend Filtering (BPTF):
    - Apply BPTF:
      python gnifti-maths.py -i input.nii -o output.nii --bptf 0.1 0.1

    Data Types:
    - Specify input data type (float or double):
      python gnifti-maths.py -i input.nii -o output.nii --dt float

    - Specify output data type (float or double):
      python gnifti-maths.py -i input.nii -o output.nii --odt double
    """
    print(examples)

def main():
    parser = argparse.ArgumentParser(description="NIFTI Toolbox for GPU-accelerated image processing using PyTorch.")
    
    # Examples argument
    parser.add_argument('--examples', action='store_true', help="Show example commands and usage.")
    
    args, unknown = parser.parse_known_args()
    
    if args.examples:
        print_examples()
        return
    
    parser.add_argument('-i', '--input', type=str, required=True, help="Path to the input NIFTI file.")
    parser.add_argument('-o', '--output', type=str, required=True, help="Path to the output NIFTI file.")
    
    # Basic operations
    parser.add_argument('--add', type=str, help="Add a scalar or another NIFTI image to the input image.")
    parser.add_argument('--sub', type=str, help="Subtract a scalar or another NIFTI image from the input image.")
    parser.add_argument('--mul', type=str, help="Multiply the input image by a scalar or another NIFTI image.")
    parser.add_argument('--div', type=str, help="Divide the input image by a scalar or another NIFTI image.")
    
    # Masking and thresholding
    parser.add_argument('--mas', type=str, help="Apply a binary mask to the input image (provide mask file path).")
    parser.add_argument('--thr', type=float, help="Apply a lower threshold to the input image.")
    parser.add_argument('--uthr', type=float, help="Apply an upper threshold to the input image.")
    
    # Advanced operations
    parser.add_argument('--exp', action='store_true', help="Apply the exponential function to the input image.")
    parser.add_argument('--log', action='store_true', help="Apply the logarithm function to the input image.")
    parser.add_argument('--sin', action='store_true', help="Apply the sine function to the input image.")
    parser.add_argument('--cos', action='store_true', help="Apply the cosine function to the input image.")
    parser.add_argument('--tan', action='store_true', help="Apply the tangent function to the input image.")
    parser.add_argument('--asin', action='store_true', help="Apply the inverse sine function to the input image.")
    parser.add_argument('--acos', action='store_true', help="Apply the inverse cosine function to the input image.")
    parser.add_argument('--atan', action='store_true', help="Apply the inverse tangent function to the input image.")
    parser.add_argument('--sqr', action='store_true', help="Square the input image.")
    parser.add_argument('--sqrt', action='store_true', help="Apply the square root function to the input image.")
    parser.add_argument('--recip', action='store_true', help="Apply the reciprocal function to the input image.")
    parser.add_argument('--abs', action='store_true', help="Apply the absolute value function to the input image.")
    parser.add_argument('--binarize', action='store_true', help="Binarize the input image (values > 0 become 1, otherwise 0).")
    parser.add_argument('--binv', action='store_true', help="Invert the binarization of the input image (values == 0 become 1, otherwise 0).")
    
    # Dilation and erosion
    parser.add_argument('--dilM', type=int, nargs=3, help="Apply mean dilation with a specified kernel size (3 integers).")
    parser.add_argument('--dilD', type=int, nargs=3, help="Apply modal dilation with a specified kernel size (3 integers).")
    parser.add_argument('--ero', type=int, nargs=3, help="Apply erosion with a specified kernel size (3 integers).")
    
    # TFCE and tensor decomposition
    parser.add_argument('--tfce', type=float, nargs=2, help="Apply TFCE (Threshold-Free Cluster Enhancement) with H and E parameters.")
    parser.add_argument('--tensor_decomp', action='store_true', help="Perform tensor decomposition and return Fractional Anisotropy (FA) image.")
    
    # BPTF (Bayesian Polynomial Trend Filtering)
    parser.add_argument('--bptf', type=float, nargs=2, help="Apply BPTF with the specified spatial and temporal smoothing parameters.")
    
    # Data types
    parser.add_argument('--dt', type=str, choices=['float', 'double'], default='float', help="Data type for the input image (default: float).")
    parser.add_argument('--odt', type=str, choices=['float', 'double'], default='float', help="Data type for the output image (default: float).")
    
    args = parser.parse_args(unknown, namespace=args)
    
    try:
        # Load input image
        img_data, affine, header = load_nifti(args.input)
        
        # Convert to specified input datatype
        img_data = img_data.to(torch.float32 if args.dt == 'float' else torch.float64)
        
        # Perform basic operations
        for op in ['add', 'sub', 'mul', 'div']:
            if getattr(args, op):
                op_data = getattr(args, op)
                if op_data.endswith('.nii') or op_data.endswith('.nii.gz'):
                    op_data = load_nifti(op_data)[0]
                else:
                    op_data = float(op_data)
                img_data = globals()[op](img_data, op_data)
        
        # Perform masking and thresholding
        if args.mas:
            mask_data = load_nifti(args.mas)[0]
            img_data = mas(img_data, mask_data)
        if args.thr is not None:
            img_data = thr(img_data, args.thr)
        if args.uthr is not None:
            img_data = uthr(img_data, args.uthr)
        
        # Perform advanced operations
        for op in ['exp', 'log', 'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'sqr', 'sqrt', 'recip', 'abs', 'binarize', 'binv']:
            if getattr(args, op):
                img_data = globals()[op](img_data)
        
        # Handle dilation and erosion
        kernel = None
        if args.dilM:
            kernel = create_kernel(*args.dilM)
            img_data = mean_dilation(img_data, kernel)
        if args.dilD:
            kernel = create_kernel(*args.dilD)
            img_data = modal_dilation(img_data, kernel)
        if args.ero:
            kernel = create_kernel(*args.ero)
            img_data = erosion(img_data, kernel)
        
        # Perform TFCE and tensor decomposition
        if args.tfce:
            img_data = tfce(img_data, *args.tfce)
        if args.tensor_decomp:
            decomp_result = tensor_decomposition(img_data)
            img_data = decomp_result[3]  # Return FA by default
        if args.bptf:
            img_data = bptf(img_data, *args.bptf)
        
        # Convert to specified output datatype
        img_data = img_data.to(torch.float32 if args.odt == 'float' else torch.float64)
        
        # Save output image
        save_nifti(img_data, affine, header, args.output)
        
        logger.info(f"Processing complete. Output saved to {args.output}")
    
    except NIFTIToolboxError as e:
        logger.error(f"NIFTI Toolbox error: {str(e)}")
        exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        exit(1)

if __name__ == '__main__':
    main()
