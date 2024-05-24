# functions for use in Conesa_krueger_jl.ipynb
# John Ryan, October 2023


#keyword-enabled structure to hold model primitives
@with_kw struct Primitives
    β::Float64 = 0.97        # discount rate
    γ::Float64 = .42         # Cobb Douglas consumption weight
    σ::Float64 = 2.0         # CRRA utility
    α::Float64 = .36         # capital share
    δ::Float64 = 0.06        # depreciation rate
    N::Int64 = 66            # death age
    Jʳ::Int64 = 46           # age of retirement
    n::Float64 = .011        # population growth rate
    a_min::Float64 = 0.001   # assets lower bound
    a_max::Float64 = 30.0    # assets upper bound
    na::Int64 = 1000         # number of asset grid points
    a_grid::Array{Float64,1} = collect(range(a_min, length = na, stop = a_max))     #asset grid
    z_grid::Vector{Float64} = [3.0, 0.5]                                            # productivity shocks
    nz::Int64 = length(z_grid)                                                      # number of productivity shocks
    initial_dist::Vector{Float64} = [0.2037, 0.7963]                                # initial distribution of productivity
    π::Matrix{Float64} = [0.9261 0.0739;                                            # Stochastic process for employment
                          0.0189 0.9811] 
    θ::Float64 = 0.11                                                               # social security tax rate
    η::Vector{Float64} = ef
end





#structure that holds model results
@with_kw mutable struct Results
    Vʳ::Matrix{Float64}         # value function retired
    gʳ::Matrix{Float64}         # policy function retired
    uʳ::Matrix{Float64}         # utility function retired
    Vʷ::Array{Float64, 3}       # value function worker
    gʷ::Array{Float64, 3}       # policy function worker
    uʷ::Array{Float64, 3}       # utility function worker
    l::Array{Float64, 3}        # labor supply
    r::Float64                  # Equilibrium interest rate
    w::Float64                  # Equilibrium wage
    b::Float64                  # retirement benefits
    Fʳ::Matrix{Float64}         # stationary dist for retired
    Fʷ::Array{Float64, 3}       # stationary dist for worker
end

function Initialize(prim::Primitives)
    # Initialize value function and policy function
    Vʳ = zeros(prim.N - prim.Jʳ, prim.na)       # retired value function (matrix)
    gʳ = zeros(prim.N - prim.Jʳ, prim.na)       # retired policy function (matrix)
    uʳ = zeros(prim.N - prim.Jʳ, prim.na)       # retired utility (matrix)
    Vʷ = zeros(prim.Jʳ-1, prim.na, prim.nz)     # worker value function (tensor)
    gʷ = zeros(prim.Jʳ-1, prim.na, prim.nz)     # worker policy function (tensor)
    uʷ = zeros(prim.Jʳ-1, prim.na, prim.nz)     # worker utility (tensor)
    l = zeros(prim.Jʳ-1, prim.na, prim.nz)      # labor supply (tensor)
    r = 0.05                                    # initial guess for interest rate
    w = 1.05                                    # initial guess for wage
    b = 0.2                                     # initial guess for retirement benefits
    Fʳ = zeros(prim.N - prim.Jʳ, prim.na)       # retired wealth distribution (matrix)
    Fʷ = zeros(prim.Jʳ-1, prim.na, prim.nz)     # worker wealth distribution (tensor)
    return Results(Vʳ, gʳ, uʳ, Vʷ, gʷ, uʷ, l, r, w, b, Fʳ, Fʷ)
end




function V_induction(prim::Primitives, res::Results)
    @unpack_Primitives prim
    @unpack_Results res
    
    u(c, l) = (c^γ * (1-l)^(1-γ))^(1-σ) / (1-σ)

    # initialize with age N
    for (a_index, a) in enumerate(a_grid)
        res.Vʳ[N-Jʳ,a_index] = u((1+r)*a + b,0)
        res.uʳ[N-Jʳ,a_index] = u((1+r)*a + b,0)
    end

    # Value function iteration for retired first
    for j in (N-Jʳ-1):-1:1  # age = j + J, iterate backwards
        for (a_index, a) in enumerate(a_grid)
            candidate_max = -Inf #bad candidate max
            budget = (1+r)*a + b #budget
            
            # perform grid search for a'
            for (ap_index, ap) in enumerate(a_grid)                 # loop over possible selections of a'
                c = budget - ap                                     # consumption given k' selection
                if c >= 0                                           # check for positivity
                    val = u(c,0) + β * res.Vʳ[j+1, ap_index]         # compute value function
                    if val>candidate_max                            # check for new max value
                        candidate_max = val                         # update max value
                        res.gʳ[j, a_index] = ap                     # update policy function
                        res.uʳ[j, a_index] = u(c,0)                 # update utility function
                    end
                end
            end
            res.Vʳ[j, a_index] = candidate_max                      # update value function
        end
    end

    # initialize with age Jʳ
    for (z_index, z) in enumerate(z_grid)
        e_zη = z * η[end]

        for (a_index, a) in enumerate(a_grid)
            candidate_max = -Inf                                              # bad candidate max
            
            # grid search
            for (ap_index, ap) in enumerate(a_grid)
                l = (γ*(1-θ)*e_zη*w - (1-γ)*(a*(1+r)-ap))/((1-θ)*w*e_zη)      # labor supply given a'
                l = max(0, min(1, l))                                          # enforce bounds
                c = (1+r)*a + w*e_zη*(1-θ)*l - ap                             # consumption given a'
                if c >= 0                                                   # check for positivity
                    val = u(c,l) + β*res.Vʳ[1,ap_index]                             # compute value function
                    if val>candidate_max                                      # check for new max value
                        candidate_max = val                                   # update max value
                        res.gʷ[end, a_index, z_index] = ap                       # update policy function
                        res.l[end, a_index, z_index] = l                         # update labor supply
                        res.uʷ[end, a_index, z_index] = u(c,l)                   # update utility function
                    end
                end
            end
            res.Vʷ[end, a_index, z_index] = candidate_max                    # update value function
        end
    end

    # Value function iteration for workers
    for j in (Jʳ-2):-1:1                                                      # age
        for (z_index, z) in enumerate(z_grid)
            e_zη = z * η[j]
            
            for (a_index, a) in enumerate(a_grid)
                candidate_max = -Inf                                                 # bad candidate max

                # grid search
                for (ap_index, ap) in enumerate(a_grid)                              # loop over vals of a'
                    l = (γ*(1-θ)*e_zη*w - (1-γ)*(a*(1+r)-ap))/((1-θ)*w*e_zη)         # labor supply given a'
                    l = max(0, min(1, l))
                    c = (1+r)*a + w*e_zη*(1-θ)*l - ap                                # consumption given a'

                    if c >= 0                                    # check for positivity
                        val = u(c,l) + β*(π[z_index,:]⋅Vʷ[j+1, ap_index, :])        # compute value function
                        if val>candidate_max                                         # check for new max value
                            candidate_max = val                                      # update max value
                            res.gʷ[j, a_index, z_index] = ap                         # update policy function
                            res.l[j, a_index, z_index] = l                           # update labor supply
                            res.uʷ[j, a_index, z_index] = u(c,l)                     # update utility function
                        end
                    end
                end
                res.Vʷ[j, a_index, z_index] = candidate_max                          # update value function

            end
        end
    end
end



function steady_dist(prim::Primitives, res::Results) 
    @unpack_Primitives prim
    @unpack_Results res

    res.Fʳ = zeros(N-Jʳ, na)
    res.Fʷ = zeros(Jʳ-1, na, nz)

    μⱼ = ones(N)
    for i in 2:N
        μⱼ[i] = μⱼ[i-1] /(1+n)
    end
    μⱼ = μⱼ/sum(μⱼ)

    # take initial dist of productivity and age 1
    res.Fʷ[1,1,:] = prim.initial_dist * μⱼ[1]

    # iterate F forward through age using policy functions for workers
    for j in 2:(Jʳ-1)
        for z_index in eachindex(z_grid)
            for a_index in eachindex(a_grid)
                for zp_index in eachindex(z_grid)
                    for (ap_index, ap) in enumerate(a_grid)
                        if ap == res.gʷ[j-1, a_index, z_index]
                            res.Fʷ[j, ap_index, zp_index] += res.Fʷ[j-1, a_index, z_index] * π[z_index, zp_index] / (1+n)
                        end
                    end
                end
            end
        end
    end

    # take dist of initial retired from last period of employed
    for a_index in eachindex(a_grid)
        for (ap_index, ap) in enumerate(a_grid)
            for z_index in eachindex(z_grid)
                if ap == res.gʷ[Jʳ-1, a_index, z_index]
                    res.Fʳ[1, a_index] += res.Fʷ[Jʳ-1, a_index, z_index] / (1+n)
                end
            end
        end
    end

    # iterate F forward through age using policy functions for retired
    for j in 2:(N-Jʳ)
        for a_index in eachindex(a_grid)
            for (ap_index, ap) in enumerate(a_grid)
                if ap == res.gʳ[j-1, a_index]
                    res.Fʳ[j, ap_index] += res.Fʳ[j-1, a_index] / (1+n)
                end
            end
        end
    end

    # renormalize to reduce numerical error
    res.Fʳ /= (sum(res.Fʳ) + sum(res.Fʷ))
    res.Fʷ /= (sum(res.Fʳ) + sum(res.Fʷ));
end




function K_L(prim::Primitives, res::Results)
    # compute capital and labor supply implied by household decisions
    @unpack_Primitives prim
    @unpack_Results res

    # compute capital and labor
    K = 0.0
    L = 0.0
    for j in 1:(Jʳ-1) # workers
        for (a_index, a) in enumerate(a_grid)
            for (z_index, z) in enumerate(z_grid)
                K += Fʷ[j, a_index, z_index] * a
                L += Fʷ[j, a_index, z_index] * z * η[j] * l[j, a_index, z_index]
            end
        end
    end

    for j in 1:(N-Jʳ)
        for (a_index, a) in enumerate(a_grid)
            K += Fʳ[j, a_index] * a
        end
    end

    return K, L
end

function market_clearing(prim::Primitives, res::Results; tol = .0001, max_iter = 200, K₀=3.32, L₀=0.34)
    # solve for the steady state using initial guess of capital and labor

    @unpack_Primitives prim
    @unpack_Results res

    K, L = (K₀, L₀)
    K_new, L_new = K_L(prim, res)
    n = 0
    error = max(abs(K_new - K), abs(L_new - L))
    while error > tol
        
        res.r = α * (L/K)^(1-α) - δ
        res.w = (1-α) * (K/L)^α
        res.b = θ * res.w * L / sum(res.Fʳ)
        V_induction(prim, res)
        steady_dist(prim, res)
        K_new, L_new = K_L(prim, res)
        K = .98*K + .02*K_new
        L = .98*L + .02*L_new
        println("K = $K, L = $L")
        n += 1
        if n > max_iter
            println("No convergence")
            break
        end
    end

    if n < max_iter
        println("Converged in $n iterations")
    end
end