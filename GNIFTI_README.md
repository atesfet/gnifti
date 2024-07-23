
# GPU-Accelerated NIFTI Image Processing Toolbox (GNIFTI)

Requires a CUDA GPU.

## GNIFTI-MATHS
usage: gnifti-maths.py [-h] [--examples] [-i INPUT] [-o OUTPUT]
        [--add ADD] [--sub SUB] [--mul MUL] [--div DIV] [--mas MAS] [--thr THR] [--uthr UTHR]
        [--exp] [--log] [--sin] [--cos] [--tan] [--asin] [--acos] [--atan] [--sqr] [--sqrt]
        [--recip] [--abs] [--bin] [--binv] [--dilM DILM DILM DILM] [--dilD DILD DILD DILD]
        [--ero ERO ERO ERO] [--tfce TFCE TFCE] [--tensor_decomp] [--bptf BPTF BPTF]
        [--dt {float,double}] [--odt {float,double}]
NIFTI Maths Toolbox
options:
 -h, --help      show this help message and exit
 --examples      Show usage examples
 -i INPUT, --input INPUT
            Input NIFTI file
 -o OUTPUT, --output OUTPUT
            Output NIFTI file
 --add ADD       Add a scalar or another NIFTI image to the input image
 --sub SUB       Subtract a scalar or another NIFTI image from the input image
 --mul MUL       Multiply the input image by a scalar or another NIFTI image
 --div DIV       Divide the input image by a scalar or another NIFTI image
 --mas MAS       Apply a binary mask to the input image (provide mask file path)
 --thr THR       Apply a lower threshold to the input image
 --uthr UTHR      Apply an upper threshold to the input image
 --exp         Apply the exponential function to the input image
 --log         Apply the logarithm function to the input image
 --sin         Apply the sine function to the input image
 --cos         Apply the cosine function to the input image
 --tan         Apply the tangent function to the input image
 --asin        Apply the inverse sine function to the input image
 --acos        Apply the inverse cosine function to the input image
 --atan        Apply the inverse tangent function to the input image
 --sqr         Square the input image
 --sqrt        Apply the square root function to the input image
 --recip        Apply the reciprocal function to the input image
 --abs         Apply the absolute value function to the input image
 --bin         Binarize the input image (values > 0 become 1, otherwise 0)
 --binv        Invert the binarization of the input image (values == 0 become 1, otherwise 0)
 --dilM DILM DILM DILM
            Apply mean dilation with a specified kernel size (3 integers)
 --dilD DILD DILD DILD
            Apply modal dilation with a specified kernel size (3 integers)
 --ero ERO ERO ERO   Apply erosion with a specified kernel size (3 integers)
 --tfce TFCE TFCE   Apply TFCE (Threshold-Free Cluster Enhancement) with H and E parameters
 --tensor_decomp    Perform tensor decomposition and return Fractional Anisotropy (FA) image
 --bptf BPTF BPTF   Apply BPTF with the specified spatial and temporal smoothing parameters
 --dt {float,double}  Data type for the input image (default: float)
 --odt {float,double} Data type for the output image (default: float)

## GNIFTI STATS
usage: gnifti-stats.py [-h] [--examples] [-i INPUT] [-o OUTPUT] [-t] [-K indexMask] [-l L] [-u U] [-r] [-R] [-e] [-E] [-v] [-V] [-m] [-M] [-s]
            [-S] [-w] [-x] [-X] [-c] [-C] [-p P] [-P P] [-a] [-n] [-k mask] [-d image] [--histogram nbins]
            [--histogram_limited nbins min max]
NIFTI Statistics Toolbox
options:
 -h, --help      show this help message and exit
 --examples      Show usage examples
 -i INPUT, --input INPUT
            Input NIFTI file
 -o OUTPUT, --output OUTPUT
            Output NIFTI file
 -t          Separate output line for each 3D volume of a 4D timeseries
 -K indexMask     Generate separate submasks from indexMask
 -l L         Set lower threshold
 -u U         Set upper threshold
 -r          Output robust min and max intensity
 -R          Output min and max intensity
 -e          Output mean entropy
 -E          Output mean entropy of nonzero voxels
 -v          Output voxels and volume
 -V          Output voxels and volume for nonzero voxels
 -m          Output mean
 -M          Output mean for nonzero voxels
 -s          Output standard deviation
 -S          Output standard deviation for nonzero voxels
 -w          Output smallest ROI containing nonzero voxels
 -x          Output coordinates of maximum voxel
 -X          Output coordinates of minimum voxel
 -c          Output centre-of-gravity in mm coordinates
 -C          Output centre-of-gravity in voxel coordinates
 -p P         Output nth percentile
 -P P         Output nth percentile for nonzero voxels
 -a          Use absolute values of all image intensities
 -n          Treat NaN or Inf as zero
 -k mask        Use specified image for masking
 -d image       Take difference between base image and specified image
 --histogram nbins   Output histogram with nbins
 --histogram_limited nbins min max
            Output histogram with nbins and limits
