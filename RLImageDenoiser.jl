## imports

using Images
using ReinforcementLearning
using FileIO
using Noise
using Pipe
using Combinatorics
using BenchmarkTools

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
@time orig_imgs = load.(filenames)

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
poisson_wrapper(img::AbstractMatrix) = poisson(img, (abs(rand(Int)) % 90) + 11, clip=true)

# use random 5 to 20 levels
quantization_wrapper(img::AbstractMatrix) = quantization(img, (abs(rand(Int)) % 10) + 11)

## add random combinations of noise types to each image

# list of pointers to all the functions we want to permute
noise_funcs = [add_gauss_wrapper, mult_gauss_wrapper, salt_pepper_wrapper, poisson_wrapper, quantization_wrapper]

# this list has several nested layers
# first layer lets you select how many noise functions to apply to an image (1 to 5)
# second layer lets you select which permutation of n noise functions to apply
# third layer contains the actual functions
@time noise_permutations = collect.([permutations(noise_funcs, n) for n ∈ 1:length(noise_funcs)])

function noisify_image(img::AbstractMatrix)
    # successively apply each noise function to the image
    img_copy = copy(img)
    for func ∈ rand(rand(noise_permutations))
        img_copy = func(img_copy)
    end
    return img_copy
end

@time noisy_imgs = noisify_image.(orig_imgs)

## save the noisy images

filenames = readdir(ORIG_IMG_PATH)
filenames = NOISY_IMG_PATH .* filenames
@time save.(filenames, noisy_imgs)