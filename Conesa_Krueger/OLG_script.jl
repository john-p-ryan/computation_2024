using Plots, Parameters, LinearAlgebra, Optim, DelimitedFiles

# import functions from OLG_functions.jl
include("OLG_functions.jl")

# read in age - efficiency profile from ef.txt
ef = readdlm("Conesa_krueger/ef.txt", '\t')[:,1]

plot(ef, xlabel="Model Age", label="", ylabel="Deterministic Efficiency", 
title="Age-Efficiency Profile", lw=1.5, dpi=400)
savefig("Conesa_krueger/ef.png")


#########################################################################
# With social security
#########################################################################

prim = Primitives()
res = Initialize(prim)
res = V_induction(prim, res)
res = steady_dist(prim, res)
res = market_clearing(prim, res)




# plot some results

# value function age 50
plot(prim.a_grid, 
    res.Vʳ[end-16,:], 
    label = "V(a)", 
    title = "Value function for age 50")
xlabel!("Assets")


# value function age 20)
# plot value function for age 20
plot(prim.a_grid, res.Vʷ[20,:,1], label = "z=3.0", title = "Value function for age 20")
plot!(prim.a_grid, res.Vʷ[20,:,2], label = "z=0.5", title = "Value function for age 20")
# update x and y labels
xlabel!("Assets")
ylabel!("V(a)")


# plot policy function for age 20 for both z
plot(prim.a_grid, res.aʷ[20,:,1], label = "z = 3", title = "Policy function for age 20")
plot!(prim.a_grid, res.aʷ[20,:,2], label = "z = 0.5")
# change x label
xlabel!("Assets")
# change y label
ylabel!("a'(a)")


Fʷ_age_assets = sum(res.Fʷ, dims = 3)[:,:,1]
# join Fʷ_age_assets with Fʳ on age
F_combined = vcat(Fʷ_age_assets, res.Fʳ)
F_collapsed = sum(F_combined, dims = 1)[1,:]
Fʷ_collapsed = sum(Fʷ_age_assets, dims = 1)[1,:]
Fʳ_collapsed = sum(res.Fʳ, dims = 1)[1,:]


function histogram_from_pmf(M, x_grid, k)
    """
    Generates a histogram from a nonlinear probability mass function (PMF).
  
    Args:
      M: An array representing the PMF values.
      x_grid: An array of evenly spaced points corresponding to the PMF.
      k: The desired number of bins in the histogram.
  
    Returns:
      A tuple containing:
        - bin_centers: The centers of the histogram bins.
        - bin_heights: The corresponding heights (probabilities) of the bins.
    """
  
    # 1. Determine bin edges
    bin_width = (x_grid[end] - x_grid[1]) / k
    bin_edges = range(x_grid[1] - bin_width / 2, stop=x_grid[end] + bin_width / 2, length=k + 1)
  
    # 2. Initialize bin heights
    bin_heights = zeros(k)
  
    # 3. Assign PMF values to bins
    for i in eachindex(M)
      # Find the corresponding bin index
      bin_index = searchsortedfirst(bin_edges, x_grid[i]) - 1 
      
      # Handle edge cases (points exactly on bin edges)
      if bin_index < 1
        bin_index = 1
      elseif bin_index > k
        bin_index = k
      end
  
      bin_heights[bin_index] += M[i]
    end
  
    # 4. Normalize bin heights to represent probabilities
    bin_heights ./= sum(M)
  
    # 5. Calculate bin centers
    bin_centers = [0.5 * (bin_edges[i] + bin_edges[i+1]) for i in 1:k]
    
    return bin_centers, bin_heights
end

bins, heights = histogram_from_pmf(F_collapsed, prim.a_grid, 30)
bar(bins, heights, label="Wealth Distribution", xlabel="Wealth", ylabel="Density", title="Wealth Distribution", legend=:topleft)


function lorenz_curve(a_grid, F_collapsed)
    # Calculate the cumulative sum of wealth and population
    cumulative_wealth = cumsum(a_grid .* F_collapsed)
    cumulative_population = cumsum(F_collapsed)

    # Normalize to get percentages
    total_wealth = cumulative_wealth[end]
    total_population = cumulative_population[end]
    percent_wealth = cumulative_wealth / total_wealth
    percent_population = cumulative_population / total_population

    # Add (0,0) point for plotting
    percent_wealth = [0; percent_wealth]
    percent_population = [0; percent_population]

    return percent_population, percent_wealth
end

percent_population, percent_wealth = lorenz_curve(prim.a_grid, F_collapsed)


# Plot the Lorenz curve
plot(percent_population, percent_wealth,
        xlabel="Cumulative Share of Population",
        ylabel="Cumulative Share of Wealth",
        title="Lorenz Curve",
        label="Lorenz Curve",
        legend=:topleft,
        xlims=(0, 1),
        ylims=(0, 1),
        linewidth=2,
        linecolor=:blue)

# Add a 45-degree line for perfect equality
plot!([0, 1], [0, 1], label="Perfect Equality",
    linewidth=2, linestyle=:dash, linecolor=:red)


function gini_from_lorenz(percent_population, percent_wealth)
    n = length(percent_population)
    area_under_lorenz = 0.0
    for i in 2:n
        area_under_lorenz += 0.5 * (percent_population[i] - percent_population[i-1]) * (percent_wealth[i] + percent_wealth[i-1])
    end
    gini = 1.0 - 2.0 * area_under_lorenz
    return gini
end

gini = gini_from_lorenz(percent_population, percent_wealth)


function gini_by_age(prim, Fʷ, Fʳ)
    gini_age = zeros(prim.N)
    Fʷ_age_assets = sum(Fʷ, dims = 3)[:,:,1]
    F_combined = vcat(Fʷ_age_assets, Fʳ)
    for i in 1:prim.N
        percent_pop, percent_wealth = lorenz_curve(prim.a_grid, F_combined[i,:])
        gini_age[i] = gini_from_lorenz(percent_pop, percent_wealth)
    end
    return gini_age[2:end]
end

gini_age = gini_by_age(prim, res.Fʷ, res.Fʳ)

plot(2:prim.N, gini_age, label="Gini by Age", xlabel="Age", ylabel="Gini Coefficient", title="Gini Coefficient by Age")







#########################################################################
# Without social security
#########################################################################

prim2 = Primitives(θ=0.0, a_max=40.0, na=1000)
res2 = Initialize(prim2)
res2 = V_induction(prim2, res2)
res2 = steady_dist(prim2, res2)
res2 = market_clearing(prim2, res2)

Fʷ_age_assets2 = sum(res2.Fʷ, dims = 3)[:,:,1]
F_combined2 = vcat(Fʷ_age_assets2, res2.Fʳ)
F_collapsed2 = sum(F_combined2, dims = 1)[1,:]

bins2, heights2 = histogram_from_pmf(F_collapsed2, prim2.a_grid, 30)
bar(bins2, heights2, label="Wealth Distribution", xlabel="Wealth", ylabel="Density", title="Wealth Distribution", legend=:topleft)
percent_pop2, percent_wealth2 = lorenz_curve(prim2.a_grid, F_collapsed2)

gini2 = gini_from_lorenz(percent_pop2, percent_wealth2)


# Plot the Lorenz curve
plot(percent_population, percent_wealth,
        xlabel="Cumulative Share of Population",
        ylabel="Cumulative Share of Wealth",
        title="Lorenz Curve",
        label="Lorenz Curve with SS",
        legend=:topleft,
        xlims=(0, 1),
        ylims=(0, 1),
        linewidth=2,
        linecolor=:blue)

plot!(percent_pop2, percent_wealth2, label="Lorenz Curve without SS", linewidth=2, linecolor=:green)

# Add a 45-degree line for perfect equality
plot!([0, 1], [0, 1], label="Perfect Equality",
    linewidth=2, linestyle=:dash, linecolor=:red)


gini_age2 = gini_by_age(prim2, res2.Fʷ, res2.Fʳ)

plot(2:prim.N, gini_age, label="With Social Security", 
xlabel="Model Age", ylabel="Gini Coefficient", title="Gini Coefficient by Age", lw=1.5)

plot!(2:prim2.N, gini_age2, label="Without Social Security", lw=1.5)