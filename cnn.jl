using Flux, LinearAlgebra, Polynomials, Plots

const n = 100
# target polynomial
c = ChebyshevT([-1,0,-2,0,1,0,1,2,3])
target = convert(Polynomial, c)
plot(target, (-1.,1.)...,label="target")

length(target.coeffs)

# mapping polynomial to circulant matrix
S = circshift(Matrix{Float64}(I, n, n),(1,0))
param = zeros(n)
param[1:9] = target.coeffs
Circulant = param
for k in 1:n-1
    Circulant = hcat(Circulant, S^k*param)
end

# creating dataset
bs = 3000
x = randn(Float32,n,1,bs)
y = convert(Array{Float32},reshape(transpose(Circulant)*dropdims(x;dims=2) ,(n,1,bs)))
data = [(x,y)]

# padding function to work modulo n
function pad_cycl(x;l=1,r=1)
    last = size(x,1)
    xl = selectdim(x,1,last-l+1:last)
    xr = selectdim(x,1,1:r)
    cat(xl, x, xr, dims=1)
end

# neural network with 7 convolution layers
model = Chain(
    x -> pad_cycl(x,l=0,r=2),
    CrossCor((3,),1=>1,bias=Flux.Zeros()),
    x -> pad_cycl(x,l=0,r=2),
    CrossCor((3,),1=>1,bias=Flux.Zeros()),
    x -> pad_cycl(x,l=0,r=2),
    CrossCor((3,),1=>1,bias=Flux.Zeros()),
    x -> pad_cycl(x,l=0,r=2),
    CrossCor((3,),1=>1,bias=Flux.Zeros()),
    x -> pad_cycl(x,l=0,r=2),
    CrossCor((3,),1=>1,bias=Flux.Zeros()),
    x -> pad_cycl(x,l=0,r=2),
    CrossCor((3,),1=>1,bias=Flux.Zeros()),
    x -> pad_cycl(x,l=0,r=2),
    CrossCor((3,),1=>1,bias=Flux.Zeros())
)

loss(x, y) = Flux.Losses.mse(model(x), y)
ps = Flux.params(model)
loss_vector = Vector{Float32}()
logging_loss() = push!(loss_vector, loss(x, y))
opt = ADAM(0.2)
n_epochs = 1700
for epochs in 1:n_epochs
    Flux.train!(loss, ps, data, opt, cb=logging_loss)
    if epochs % 50 == 0
        println("Epoch: ", epochs, " | Loss: ", loss(x,y))
    end
end

plot(loss_vector)

pred = Polynomial([1])
for p in ps
    if typeof(p) <: Array
        pred *= Polynomial([p...])
    end
end
plot(target, (-1.,1.)...,label="target")
ylims!((-10,10))
plot!(pred, (-1.,1.)...,label="pred")

