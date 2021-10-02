## imports

using Images
using ReinforcementLearning
using FileIO
using Noise
using Pipe
using Combinatorics

## constants

const ORIG_IMG_PATH = "./images/orig/"
const NOISY_IMG_PATH = "./images/noisy/"

## TODO

#=
1. Import images
2. Generate noisy images
    * Additive white gaussian
    * Multiplicative white gaussian
    * Salt and pepper
    * Poisson
    * Quantization
3. TODO
=#

## import images

filenames = readdir(ORIG_IMG_PATH, join=true)
orig_imgs = load.(filenames)

## declare wrapper functions

# to make functional programming easier, we want to wrap the noise functions
# so we can keep the correct kwargs,
# randomly set σ for some of them,
# and expose only one input (i.e., the image)

# use random σ (std dev) from 0 to 0.5
add_gauss_wrapper(img::AbstractMatrix) = add_gauss(img, rand() / 2, clip=true)
mult_gauss_wrapper(img::AbstractMatrix) = mult_gauss(img, rand() / 2, clip=true)
salt_pepper_wrapper(img::AbstractMatrix) = salt_pepper(img, rand() / 2)

# use random 10 to 100 photons
poisson_wrapper(img::AbstractMatrix) = poisson(img, (abs(rand(Int)) % 90) + 10)

# use random 5 to 20 levels
quantization_wrapper(img::AbstractMatrix) = quantization(img, (abs(rand(Int)) % 10) + 10)

## add random combinations of noise types to each image

