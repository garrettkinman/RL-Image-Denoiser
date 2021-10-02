## imports

using Images
using ReinforcementLearning
using FileIO
using Noise

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

test_img = load(ORIG_IMG_PATH * "00000000.jpg")

## generate noisy images

add_gauss(test_img, clip=true)
salt_pepper(test_img)
poisson(test_img, 10, clip=true)
quantization(test_img, 10)