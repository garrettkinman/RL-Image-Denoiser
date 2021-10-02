## imports

using Images
using ReinforcementLearning
using FileIO
using Noise
using Pipe

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

## generate noisy images

orig_imgs[1]
add_gauss(orig_imgs[1], clip=true)
salt_pepper(orig_imgs[1])
poisson(orig_imgs[1], 10, clip=true)
quantization(orig_imgs[1], 10)

@pipe orig_imgs[1] |> add_gauss(_, clip=true) |> salt_pepper |> poisson(_, 10, clip=true) |> quantization(_, 10)