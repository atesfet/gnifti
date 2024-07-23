import argparse
import torch
import nibabel as nib
import numpy as np
import sys
import os
import pandas as pd
from typing import Tuple, Optional, List
from multiprocessing import Pool, set_start_method

# Ensure the 'spawn' start method is used
set_start_method('spawn', force=True)

# Check if CUDA is available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_nifti(file_path: str) -> Tuple[torch.Tensor, np.ndarray]:
    try:
        nifti = nib.load(file_path)
        data = torch.tensor(nifti.get_fdata(), dtype=torch.float32, device=DEVICE)
        return data, nifti.affine
    except Exception as e:
        print(f"Error loading NIFTI file: {e}", file=sys.stderr)
        sys.exit(1)

def save_nifti(data: torch.Tensor, affine: np.ndarray, file_path: str) -> None:
    try:
        nifti = nib.Nifti1Image(data.cpu().numpy(), affine)
        nib.save(nifti, file_path)
    except Exception as e:
        print(f"Error saving NIFTI file: {e}", file=sys.stderr)
        sys.exit(1)

def apply_thresholds(data: torch.Tensor, lower: Optional[float] = None, upper: Optional[float] = None) -> torch.Tensor:
    with torch.no_grad():
        if lower is not None:
            data = torch.clamp(data, min=lower)
        if upper is not None:
            data = torch.clamp(data, max=upper)
    return data

@torch.no_grad()
def robust_range(data: torch.Tensor) -> Tuple[float, float]:
    sorted_data = torch.sort(data.flatten())[0]
    lower = sorted_data[int(0.02 * len(sorted_data))]
    upper = sorted_data[int(0.98 * len(sorted_data))]
    return lower.item(), upper.item()

@torch.no_grad()
def entropy(data: torch.Tensor) -> float:
    p = data / torch.sum(data)
    return -torch.sum(p * torch.log(p + 1e-10)).item()

@torch.no_grad()
def percentile(data: torch.Tensor, n: float) -> float:
    k = int((n / 100.0) * data.numel())
    return torch.kthvalue(data.flatten(), k)[0].item()

@torch.no_grad()
def histogram(data: torch.Tensor, nbins: int, min_val: Optional[float] = None, max_val: Optional[float] = None) -> np.ndarray:
    if min_val is None:
        min_val = torch.min(data).item()
    if max_val is None:
        max_val = torch.max(data).item()
    hist = torch.histc(data.flatten(), bins=nbins, min=min_val, max=max_val)
    return hist.cpu().numpy()

def print_examples():
    examples = """
    Examples of usage:
    ------------------
    Basic usage:
    python gnifti-stats.py -i input_file.nii -o output_file.nii

    Apply lower and upper thresholds:
    python gnifti-stats.py -i input_file.nii -o output_file.nii -l 0.1 -u 0.9

    Output robust min and max intensity:
    python gnifti-stats.py -i input_file.nii -o output_file.nii -r

    Output min and max intensity:
    python gnifti-stats.py -i input_file.nii -o output_file.nii -R

    Output mean entropy:
    python gnifti-stats.py -i input_file.nii -o output_file.nii -e

    Output mean entropy of nonzero voxels:
    python gnifti-stats.py -i input_file.nii -o output_file.nii -E

    Output voxel count and volume:
    python gnifti-stats.py -i input_file.nii -o output_file.nii -v

    Output voxel count and volume for nonzero voxels:
    python gnifti-stats.py -i input_file.nii -o output_file.nii -V

    Output mean intensity:
    python gnifti-stats.py -i input_file.nii -o output_file.nii -m

    Output mean intensity for nonzero voxels:
    python gnifti-stats.py -i input_file.nii -o output_file.nii -M

    Output standard deviation:
    python gnifti-stats.py -i input_file.nii -o output_file.nii -s

    Output standard deviation for nonzero voxels:
    python gnifti-stats.py -i input_file.nii -o output_file.nii -S

    Output smallest ROI containing nonzero voxels:
    python gnifti-stats.py -i input_file.nii -o output_file.nii -w

    Output coordinates of maximum intensity voxel:
    python gnifti-stats.py -i input_file.nii -o output_file.nii -x

    Output coordinates of minimum intensity voxel:
    python gnifti-stats.py -i input_file.nii -o output_file.nii -X

    Output center-of-gravity in mm coordinates:
    python gnifti-stats.py -i input_file.nii -o output_file.nii -c

    Output center-of-gravity in voxel coordinates:
    python gnifti-stats.py -i input_file.nii -o output_file.nii -C

    Output the nth percentile:
    python gnifti-stats.py -i input_file.nii -o output_file.nii -p 90

    Output the nth percentile for nonzero voxels:
    python gnifti-stats.py -i input_file.nii -o output_file.nii -P 90

    Use absolute values of all image intensities:
    python gnifti-stats.py -i input_file.nii -o output_file.nii -a

    Treat NaN or Inf as zero:
    python gnifti-stats.py -i input_file.nii -o output_file.nii -n

    Use specified image for masking:
    python gnifti-stats.py -i input_file.nii -o output_file.nii -k mask_file.nii

    Calculate the difference between the base image and the specified image:
    python gnifti-stats.py -i input_file.nii -o output_file.nii -d diff_file.nii

    Output histogram with specified number of bins:
    python gnifti-stats.py -i input_file.nii -o output_file.nii --histogram 100

    Output histogram with specified number of bins and limits:
    python gnifti-stats.py -i input_file.nii -o output_file.nii --histogram_limited 100 0.0 1.0

    Process multiple files in parallel:
    python gnifti-stats.py -i input_directory -o output_directory --parallel 100

    Process multiple files in parallel with fractional batch size:
    python gnifti-stats.py -i input_directory -o output_directory --parallel quarter
    """
    print(examples)

def get_nifti_files(input_dir: str) -> List[str]:
    return [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(('.nii', '.nii.gz'))]

def process_file(file_path: str, operation: str, args: argparse.Namespace) -> Tuple[str, float]:
    data, affine = load_nifti(file_path)
    
    if args.k:
        mask, _ = load_nifti(args.k)
        data = torch.where(mask > 0, data, torch.tensor(0., device=DEVICE))
    else:
        data = apply_thresholds(data, args.l, args.u)
    
    if args.d:
        diff_data, _ = load_nifti(args.d)
        data = data - diff_data
    
    if args.a:
        data = torch.abs(data)
    
    if args.n:
        data = torch.nan_to_num(data, 0)
    
    nonzero_mask = data != 0
    
    result = None
    if operation == 'M':
        result = torch.mean(data[nonzero_mask]).item()
    elif operation == 'm':
        result = torch.mean(data).item()
    elif operation == 'S':
        result = torch.std(data[nonzero_mask]).item()
    elif operation == 's':
        result = torch.std(data).item()
    elif operation == 'e':
        result = entropy(data)
    elif operation == 'E':
        result = entropy(data[nonzero_mask])
    elif operation == 'p':
        result = percentile(data, args.p)
    elif operation == 'P':
        result = percentile(data[nonzero_mask], args.P)
    elif operation == 'r':
        result = robust_range(data)
    elif operation == 'R':
        result = [torch.min(data).item(), torch.max(data).item()]
    elif operation == 'v':
        voxel_count = torch.sum(nonzero_mask).item()
        volume = voxel_count * torch.prod(torch.tensor(data.shape[1:], device=DEVICE)).item()
        result = [voxel_count, volume]
    elif operation == 'V':
        voxel_count = torch.sum(nonzero_mask).item()
        volume = voxel_count * torch.prod(torch.tensor(data.shape[1:], device=DEVICE)).item()
        result = [voxel_count, volume]
    elif operation == 'w':
        nonzero_indices = torch.nonzero(data)
        min_coords = torch.min(nonzero_indices, dim=0)[0]
        max_coords = torch.max(nonzero_indices, dim=0)[0]
        result = (min_coords[0].item(), max_coords[0].item() - min_coords[0].item() + 1,
                  min_coords[1].item(), max_coords[1].item() - min_coords[1].item() + 1,
                  min_coords[2].item(), max_coords[2].item() - min_coords[2].item() + 1,
                  min_coords[3].item() if len(data.shape) > 3 else 0,
                  max_coords[3].item() - min_coords[3].item() + 1 if len(data.shape) > 3 else 1)
    elif operation == 'x':
        max_index = torch.argmax(data)
        result = torch.unravel_index(max_index, data.shape)
    elif operation == 'X':
        min_index = torch.argmin(data)
        result = torch.unravel_index(min_index, data.shape)
    elif operation == 'c':
        indices = torch.meshgrid(*[torch.arange(s, device=DEVICE) for s in data.shape])
        cog = torch.stack([torch.sum(i * data) / torch.sum(data) for i in indices])
        result = torch.matmul(torch.tensor(affine[:3, :3], device=DEVICE), cog[:3]).tolist()
    elif operation == 'C':
        indices = torch.meshgrid(*[torch.arange(s, device=DEVICE) for s in data.shape])
        cog = torch.stack([torch.sum(i * data) / torch.sum(data) for i in indices])
        result = cog.tolist()
    elif operation == '--histogram':
        hist = histogram(data[nonzero_mask], args.histogram)
        result = hist.tolist()
    elif operation == '--histogram_limited':
        hist = histogram(data[nonzero_mask], int(args.histogram_limited[0]), args.histogram_limited[1], args.histogram_limited[2])
        result = hist.tolist()

    torch.cuda.empty_cache()  # Clear GPU memory
    return os.path.basename(file_path), result

def process_batch(file_paths: List[str], operation: str, args: argparse.Namespace) -> List[Tuple[str, float]]:
    results = []
    for fp in file_paths:
        try:
            results.append(process_file(fp, operation, args))
        except torch.cuda.OutOfMemoryError:
            print(f"Out of memory while processing {fp}. Trying smaller batch size.")
            break  # Exit the loop to handle the memory issue
    return results

def main(args: argparse.Namespace) -> None:
    if args.examples:
        print_examples()
        sys.exit(0)

    if os.path.isdir(args.input):
        file_paths = get_nifti_files(args.input)
        num_files = len(file_paths)
        if args.parallel:
            if args.parallel in ['quarter', 'third', 'half', 'all']:
                if args.parallel == 'quarter':
                    batch_size = max(1, num_files // 4)
                elif args.parallel == 'third':
                    batch_size = max(1, num_files // 3)
                elif args.parallel == 'half':
                    batch_size = max(1, num_files // 2)
                else:  # 'all'
                    batch_size = num_files
            else:
                batch_size = int(args.parallel)
        else:
            batch_size = 1
        
        batches = [file_paths[i:i + batch_size] for i in range(0, num_files, batch_size)]
        results = []
        
        with Pool() as pool:
            for batch in batches:
                try:
                    results.extend(pool.apply_async(process_batch, (batch, args.operation, args)).get())
                except torch.cuda.OutOfMemoryError:
                    print("Out of memory. Reducing batch size and retrying.")
                    batch_size = max(1, batch_size // 2)
                    batches = [file_paths[i:i + batch_size] for i in range(0, num_files, batch_size)]
        
        df = pd.DataFrame(results, columns=['patient_ID', 'value'])
        output_file = f"{args.operation}_processed.csv"
        df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
    else:
        data, affine = load_nifti(args.input)
        
        if args.k:
            mask, _ = load_nifti(args.k)
            data = torch.where(mask > 0, data, torch.tensor(0., device=DEVICE))
        else:
            data = apply_thresholds(data, args.l, args.u)
        
        if args.d:
            diff_data, _ = load_nifti(args.d)
            data = data - diff_data
        
        if args.a:
            data = torch.abs(data)
        
        if args.n:
            data = torch.nan_to_num(data, 0)
        
        nonzero_mask = data != 0
        
        result = None
        if args.operation == 'M':
            result = torch.mean(data[nonzero_mask]).item()
        elif args.operation == 'm':
            result = torch.mean(data).item()
        elif args.operation == 'S':
            result = torch.std(data[nonzero_mask]).item()
        elif args.operation == 's':
            result = torch.std(data).item()
        elif args.operation == 'e':
            result = entropy(data)
        elif args.operation == 'E':
            result = entropy(data[nonzero_mask])
        elif args.operation == 'p':
            result = percentile(data, args.p)
        elif args.operation == 'P':
            result = percentile(data[nonzero_mask], args.P)
        elif args.operation == 'r':
            result = robust_range(data)
        elif args.operation == 'R':
            result = [torch.min(data).item(), torch.max(data).item()]
        elif args.operation == 'v':
            voxel_count = torch.sum(nonzero_mask).item()
            volume = voxel_count * torch.prod(torch.tensor(data.shape[1:], device=DEVICE)).item()
            result = [voxel_count, volume]
        elif args.operation == 'V':
            voxel_count = torch.sum(nonzero_mask).item()
            volume = voxel_count * torch.prod(torch.tensor(data.shape[1:], device=DEVICE)).item()
            result = [voxel_count, volume]
        elif args.operation == 'w':
            nonzero_indices = torch.nonzero(data)
            min_coords = torch.min(nonzero_indices, dim=0)[0]
            max_coords = torch.max(nonzero_indices, dim=0)[0]
            result = (min_coords[0].item(), max_coords[0].item() - min_coords[0].item() + 1,
                      min_coords[1].item(), max_coords[1].item() - min_coords[1].item() + 1,
                      min_coords[2].item(), max_coords[2].item() - min_coords[2].item() + 1,
                      min_coords[3].item() if len(data.shape) > 3 else 0,
                      max_coords[3].item() - min_coords[3].item() + 1 if len(data.shape) > 3 else 1)
        elif args.operation == 'x':
            max_index = torch.argmax(data)
            result = torch.unravel_index(max_index, data.shape)
        elif args.operation == 'X':
            min_index = torch.argmin(data)
            result = torch.unravel_index(min_index, data.shape)
        elif args.operation == 'c':
            indices = torch.meshgrid(*[torch.arange(s, device=DEVICE) for s in data.shape])
            cog = torch.stack([torch.sum(i * data) / torch.sum(data) for i in indices])
            result = torch.matmul(torch.tensor(affine[:3, :3], device=DEVICE), cog[:3]).tolist()
        elif args.operation == 'C':
            indices = torch.meshgrid(*[torch.arange(s, device=DEVICE) for s in data.shape])
            cog = torch.stack([torch.sum(i * data) / torch.sum(data) for i in indices])
            result = cog.tolist()
        elif args.operation == '--histogram':
            hist = histogram(data[nonzero_mask], args.histogram)
            result = hist.tolist()
        elif args.operation == '--histogram_limited':
            hist = histogram(data[nonzero_mask], int(args.histogram_limited[0]), args.histogram_limited[1], args.histogram_limited[2])
            result = hist.tolist()
        
        print(result)
        
        if args.output:
            save_nifti(data, affine, args.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NIFTI Statistics Toolbox")
    parser.add_argument("--examples", action="store_true", help="Show usage examples")
    parser.add_argument("-i", "--input", required=True, help="Input NIFTI file or directory")
    parser.add_argument("-o", "--output", help="Output NIFTI file or directory")
    parser.add_argument("-t", action="store_true", help="Separate output line for each 3D volume of a 4D timeseries")
    parser.add_argument("-K", metavar="indexMask", help="Generate separate submasks from indexMask")
    parser.add_argument("-l", type=float, help="Set lower threshold")
    parser.add_argument("-u", type=float, help="Set upper threshold")
    parser.add_argument("-r", action="store_true", help="Output robust min and max intensity")
    parser.add_argument("-R", action="store_true", help="Output min and max intensity")
    parser.add_argument("-e", action="store_true", help="Output mean entropy")
    parser.add_argument("-E", action="store_true", help="Output mean entropy of nonzero voxels")
    parser.add_argument("-v", action="store_true", help="Output voxels and volume")
    parser.add_argument("-V", action="store_true", help="Output voxels and volume for nonzero voxels")
    parser.add_argument("-m", action="store_true", help="Output mean")
    parser.add_argument("-M", action="store_true", help="Output mean for nonzero voxels")
    parser.add_argument("-s", action="store_true", help="Output standard deviation")
    parser.add_argument("-S", action="store_true", help="Output standard deviation for nonzero voxels")
    parser.add_argument("-w", action="store_true", help="Output smallest ROI containing nonzero voxels")
    parser.add_argument("-x", action="store_true", help="Output coordinates of maximum voxel")
    parser.add_argument("-X", action="store_true", help="Output coordinates of minimum voxel")
    parser.add_argument("-c", action="store_true", help="Output centre-of-gravity in mm coordinates")
    parser.add_argument("-C", action="store_true", help="Output centre-of-gravity in voxel coordinates")
    parser.add_argument("-p", type=float, help="Output nth percentile")
    parser.add_argument("-P", type=float, help="Output nth percentile for nonzero voxels")
    parser.add_argument("-a", action="store_true", help="Use absolute values of all image intensities")
    parser.add_argument("-n", action="store_true", help="Treat NaN or Inf as zero")
    parser.add_argument("-k", metavar="mask", help="Use specified image for masking")
    parser.add_argument("-d", metavar="image", help="Take difference between base image and specified image")
    parser.add_argument("--histogram", type=int, metavar="nbins", help="Output histogram with nbins")
    parser.add_argument("--histogram_limited", nargs=3, type=float, metavar=("nbins", "min", "max"), help="Output histogram with nbins and limits")
    parser.add_argument("--parallel", help="Run in parallel mode. Options: 'quarter', 'third', 'half', 'all', or an integer specifying the batch size")

    args = parser.parse_args()
    
    # Determine the operation based on the arguments provided
    if args.M:
        args.operation = 'M'
    elif args.m:
        args.operation = 'm'
    elif args.S:
        args.operation = 'S'
    elif args.s:
        args.operation = 's'
    elif args.e:
        args.operation = 'e'
    elif args.E:
        args.operation = 'E'
    elif args.p:
        args.operation = 'p'
    elif args.P:
        args.operation = 'P'
    elif args.r:
        args.operation = 'r'
    elif args.R:
        args.operation = 'R'
    elif args.v:
        args.operation = 'v'
    elif args.V:
        args.operation = 'V'
    elif args.w:
        args.operation = 'w'
    elif args.x:
        args.operation = 'x'
    elif args.X:
        args.operation = 'X'
    elif args.c:
        args.operation = 'c'
    elif args.C:
        args.operation = 'C'
    elif args.histogram:
        args.operation = '--histogram'
    elif args.histogram_limited:
        args.operation = '--histogram_limited'
    else:
        print("No valid operation specified.")
        sys.exit(1)

    main(args)
