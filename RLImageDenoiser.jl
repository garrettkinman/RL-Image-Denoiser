## imports

using Images
using FileIO
using Noise
using Pipe
using Combinatorics
using BenchmarkTools
using POMDPs
using DeepQLearning
using Flux
using Statistics

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

## declare wrapper functions for actions

# helper functions to allow use of median filter on RGB
magnitude(x::RGB) = √(float(x.r)^2 + float(x.g)^2 + (x.b)^2)
Base.isless(x::RGB, y::RGB) = magnitude(x) < magnitude(y)
Statistics.middle(x::RGB) = x

denoised_img = mapwindow(median, noisy_imgs[1], (3, 3))
typeof(noisy_imgs[1])

#= filter types
1. Gaussian (3x3, 5x5, 7x7)
2. Median (3x3, 5x5, 7x7)
=#
gauss3(img::AbstractMatrix) = imfilter(img, Kernel.gaussian((0.5, 0.5), (3, 3)), Inner())
gauss5(img::AbstractMatrix) = imfilter(img, Kernel.gaussian((0.5, 0.5), (5, 5)), Inner())
gauss7(img::AbstractMatrix) = imfilter(img, Kernel.gaussian((0.5, 0.5), (7, 7)), Inner())
median3(img::AbstractMatrix) = mapwindow(median, img, (3, 3))
median5(img::AbstractMatrix) = mapwindow(median, img, (5, 5))
median7(img::AbstractMatrix) = mapwindow(median, img, (7, 7))

## setup RL environment
mutable struct ImageDenoiseMDP <: MDP{AbstractMatrix, Symbol} # MDP{State, Action}
    # TODO: fields
    # TODO: constructor
end

# action space for the MDP
function POMDPs.actions(m::ImageDenoiseMDP)
    return [:gauss3, :gauss5, :gauss7, :median3, :median5, :median7, :stop]
end

function POMDPs.gen(m::ImageDenoiseMDP, s, a, rng::AbstractRNG)
    sp = copy(s)
    r = 0

    # state transition
    if a == :gauss3
        sp = gauss3(sp)
    elseif a == :gauss5
        sp = gauss5(sp)
    elseif a == :gauss7
        sp = gauss7(sp)
    elseif a == :median3
        sp = median3(sp)
    elseif a == :median5
        sp = median5(sp)
    elseif a == :median7
        sp = median7(sp)
    elseif a = :stop
        sp = sp
        # TODO: reward based on image distance
    end
    
    return (sp=sp, r=r) # return named tuple of next state and the granted reward
end

#= TODO

# load MDP model from POMDPModels or define your own!
mdp = SimpleGridWorld();

# Define the Q network (see Flux.jl documentation)
# the gridworld state is represented by a 2 dimensional vector.
model = Chain(Dense(2, 32), Dense(32, length(actions(mdp))))

exploration = EpsGreedyPolicy(mdp, LinearDecaySchedule(start=1.0, stop=0.01, steps=10000/2))

solver = DeepQLearningSolver(qnetwork = model, max_steps=10000, 
                             exploration_policy = exploration,
                             learning_rate=0.005,log_freq=500,
                             recurrence=false,double_q=true, dueling=true, prioritized_replay=true)
policy = solve(solver, mdp)

sim = RolloutSimulator(max_steps=30)
r_tot = simulate(sim, mdp, policy)
println("Total discounted reward for 1 simulation: $r_tot")

=#