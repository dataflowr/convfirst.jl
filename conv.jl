using LinearAlgebra, Zygote, Plots

const n = 100
S = circshift(Matrix{Float64}(I, n, n),(1,0))

function loss(W)
    norm(W*S-S*W)/norm(W)
end

function step!(W;lr=0.003)
    # computing current loss and backprop
    current_loss, back_loss = pullback(w -> loss(w),W)
    # computing gradient
    grads = back_loss(1)[1]
    # updating W 
    W .-= lr .*grads
end

W = randn(n,n)
W ./= norm(W)

heatmap(W,clims=(-0.03,0.03),legend=:none,axis=nothing)

for i=1:1000
    step!(W)
end

heatmap(W,clims=(-0.03,0.03),legend=:none,axis=nothing)