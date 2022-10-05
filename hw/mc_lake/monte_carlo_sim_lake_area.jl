### A Pluto.jl notebook ###
# v0.19.12

using Markdown
using InteractiveUtils

# ╔═╡ 640a599e-3a27-11ed-069c-f14f3b6633e9
using CairoMakie, CSV, DataFrames, Interpolations, LinearAlgebra, Statistics, JLD2, Distributions, PlutoUI

# ╔═╡ 78a8749f-c501-4e7b-bccf-3bdf4d55df92
TableOfContents()

# ╔═╡ dbf497fa-f45b-4f75-96c0-3e0a73af0203
md"# hw1: Monte Carlo simulation to compute the area of Crater lake

the task in this hw assignment is to compute the area of Crater Lake using a Monte Carlo simulation. this problem resembles the classic one of estimating $\pi$ by Monte Carlo simulation, described [here](https://en.wikipedia.org/wiki/Monte_Carlo_method#Overview).

to represent the shape of Crater Lake, I provide a matrix `X`, whose column $i$ is the spatial coordinate of a point $\mathbf{x}_i\in\mathbb{R}^2$ that lies on the boundary of the lake. the spatial units are miles. we will employ this data tracing the lake boundary to create a representation of the shape/area of the lake.
"

# ╔═╡ 2e87f1a8-b943-4317-99b1-8125e99355f2
md"
## read in, viz the raw data
🐜 download the `crater_lake_coords.jld2` file from Github [here](https://github.com/SimonEnsemble/intro_to_data_science_fall_2022/blob/main/hw/mc_lake/crater_lake_coords.jld2). use the `load` function from the `JLD2` package (see [here](https://juliaio.github.io/JLD2.jl/dev/#save-and-load-functions)) to read the data and assign the matrix stored in the `.jld2` file as a variable `X`.

!!! note
	> `JLD2` saves and loads Julia data structures in a format comprising a subset of HDF5. -[JLD2 docs](https://github.com/JuliaIO/JLD2.jl)

	> Hierarchical Data Format (HDF) is a set of file formats (HDF4, HDF5) designed to store and organize large amounts of data. -[Wikipedia](https://en.wikipedia.org/wiki/Hierarchical_Data_Format)

"

# ╔═╡ 4f83121f-4b98-4854-adb4-7e11be25b482
X = load("./crater_lake_coords.jld2")["X"]

# ╔═╡ 60e98df1-e21a-4c8d-ba54-50455b61379c
md"🐜 use `Makie` to draw a scatter plot of the points tracing the lake (contained in `X`). 
* make the points blue, like lake water
* include x- and y-labels (well, x₁ and x₂ here) with units of miles
* include a title
* pass `aspect=DataAspect()` to `Axis` to ensure proper scaling. see [here](https://docs.makie.org/v0.17.13/examples/blocks/axis/index.html#controlling_axis_aspect_ratios).
I wrote a function `viz_lake(X)` to do this, but you don't have to.

!!! hint
	use `Makie.jl` to make a `scatter` plot. see [here](https://docs.makie.org/v0.17.13/tutorials/basic-tutorial/#scatter_plot).
"

# ╔═╡ 68ad49c5-6bd5-460c-b404-33e52292dd6d
function drawlake()
	f = Figure()
	ax = Axis(f[1, 1],
	    title = "Crater Lake",
		titlesize = 24,
	    xlabel = "Miles",
	    ylabel = "Miles",
		aspect=DataAspect()
	)
	
	return f
end

# ╔═╡ e9dbfd3f-a0d9-4dec-899b-aa62be5790dd
function drawlake(data)
	f = drawlake()
	scatter!(data[begin, :], data[end, :], label="Lake Coords")
	
	return f
end

# ╔═╡ 010f8cc2-dbc9-4fa0-bc0a-2c088bfc00d5
function drawlake(data, interp_data)
	f = drawlake(data)
	x, y = interp_data
	lines!(x, y)

	return f
end

# ╔═╡ fd1a78dc-2fb4-48f9-a61c-fa00ecfd367b
drawlake(X)

# ╔═╡ 4ec36505-66d9-4cd0-a4c5-e3718d1a5cba
md"
## centering the data
it will be convenient to center the coordinates.

🐜 compute the \"center\" of the set of points tracing the lake as the mean coordinate in the matrix `X`. assign it to be a variable `x̄`.

i.e.
```math
	\bar{\mathbf{x}} = \frac{1}{N}\sum_{i=1}^N \mathbf{x}_i
```
with $\mathbf{x}_i\in\mathbb{R}^2$ one of $N$ coordinates on the lake boundary.

!!! hint
	use the `mean` function and the `dims` keyword argument. see [docs](https://docs.julialang.org/en/v1/stdlib/Statistics/#Statistics.mean).
"

# ╔═╡ b45e7b0a-8644-4210-bec9-8fd3076b08f4
x̄ = mean(X; dims=2)

# ╔═╡ 3a088002-b238-46eb-86f8-483ca5b9a7c2
md"🐜 define a centered coordinate matrix `X̂` whose column $i$ gives the centered coordinate:
```math
	\mathbf{\hat{x}}_i := \mathbf{x}_i - \bar{\mathbf{x}}_i.
```
i.e., subtract the mean vector from each column of the matrix `X` to obtain `X̂`. this is a simple shift that preserves distances.
"

# ╔═╡ bc825fb1-5ee5-4d39-9ec2-a99ca8b68e79
X̂ = X .- x̄

# ╔═╡ 89dde6a1-1230-4841-8fca-c2563a54fd4a
md"🐜 redraw the lake boundary (exactly as above) using the centered coordinates. the lake should be the exact same shape, but centered at the origin $(0, 0)$."

# ╔═╡ a9cf8d6e-1607-44e6-8f74-769ce7bff060
drawlake(X̂)

# ╔═╡ 9a6aa8a8-fd6a-4928-bf28-b7444b2e6d3b
md"## parameterizing the lake as $r=r(\theta)$

we need a continuous representation of the shape of the lake boundary.

to achieve this, let us parameterize the lake boundary as a function $r=r(\theta)$ with $\theta \in [-\pi, \pi]$ the angle a point on the lake boundary makes with the x₁-axis and $r$ the distance of that point from the origin. i.e., we will represent the lake boundary using a parameterization in polar coordinates. 
"

# ╔═╡ e5079f42-2137-459f-a351-ddc22864085a
md"### translate the data into polar coordinates
first, let's translate the data, currently in Cartesian coordinates, into polar coordinates. 

🐜 create a vector `r` whose entry $i$ gives the distance of coordinate $\mathbf{x}_i$ from the origin $(0,0)$.
"

# ╔═╡ 0edc2e54-6552-4a56-aa03-87ae8886e826
begin
	xdim, ydim = size(X̂)
	r = zeros(ydim)
	for i=1:ydim
		x, y = X̂[:, i] # grab column
		r[i] = √(x^2 + y^2)
	end
	r
end

# ╔═╡ d866aace-c019-4fe3-a77e-d681e1af001f
md"🐜 create a vector θ whose entry $i$ gives the angle coordinate $\mathbf{x}_i$ makes with the x₁-axis.

!!! hint
	use the `atan` function [here](https://docs.julialang.org/en/v1/base/math/#Base.atan-Tuple{Number}).
"

# ╔═╡ 6d01e662-761f-4f33-89d9-7d29c4196366
begin
	θ = zeros(ydim)
	for i=1:ydim
		x, y = X̂[:, i] # grab column
		θ[i] = atan(y, x)
	end
	θ
end

# ╔═╡ cbce6c42-b5db-4b59-8399-495289c3b247
md"🐜 to check your representation of the data in polar cooradinates, plot the points tracing the lake boundary again but compute the $(x_1, x_2)$ Cartesian coordinates for `scatter` using the formulas $x_1=r \cos(\theta)$ and $x_2=r\sin(\theta)$ and your vectors `r` and `θ`.

!!! note
	this process of checking your code every step of the way is a good coding practice.
"

# ╔═╡ d3a6961d-3f70-4b9e-8bc9-38e8ad91c952
begin
	checkdata_x = r .* cos.(θ)
	checkdata_y = r .* sin.(θ)
	checkdata = hcat(checkdata_x, checkdata_y)
	println(checkdata' .- X̂) # should be zeros
	drawlake(checkdata') # plot
end

# ╔═╡ d80c35e7-ee90-464e-b7b4-6e2250efc3ff
md"### creating a continuous representation $r=r(\theta)$
the goal now is to use linear interpolation of the data $(\theta_i, r_i)$ to construct a continuous function $r(\theta)$ for $\theta \in [-\pi, \pi]$ that traces the lake boundary. we will employ the `linear_interpolation` function [here](https://juliamath.github.io/Interpolations.jl/stable/#Example-Usage) from `Interpolations.jl`.
"

# ╔═╡ c11fa2f9-a440-4b0e-85ba-27c749f4fccb
md"🐜 the `linear_interpolation` function requires the $\theta_i$ in the `θ` vector to be sorted (note, they currently are not). sort the values in `θ`. of course, the values in `r` must be permuted the same way, to preserve the pairing between the entries in `r` and `θ`. assign the sorted versions as variables `r_sorted` and `θ_sorted`.

!!! hint
	use the `sortperm` function, then slice both `θ` and `r`. see [here](https://docs.julialang.org/en/v1/base/sort/#Base.sortperm).
"

# ╔═╡ b4addc63-8f97-4c15-8e86-e0f5733071ef
p = sortperm(θ)

# ╔═╡ 95d4cfa4-964d-45f0-8496-6a7beb382072
θ_sorted = θ[p]

# ╔═╡ 6fe5a3de-ddda-426a-a072-1db6892f22aa
r_sorted = r[p]

# ╔═╡ d4a35029-ff51-41f4-ab80-e9e61455d0c1
md"the function $r(\theta)$ is periodic because $r(-\pi)=r(\pi)$. the `linear_interpolation` function can represent a periodic function. however, it requires us to repeat either the first or last data point by copying it onto the other side of the periodic boundary.

🐜 with $N$ the index of the last data point in the sorted arrays, prepend the array `θ_sorted` with $\theta_N-2\pi$ and the array `r_sorted` with $r_N$.
"

# ╔═╡ d9849427-3389-4040-900b-d4eadc922c7d
N = length(θ)

# ╔═╡ 4211a63e-c894-45fc-bb38-b13850dcae4d
prepend!(θ_sorted, [θ_sorted[N] - 2π])

# ╔═╡ ef735579-310e-4c2d-91f0-f7ac99d71cf0
prepend!(r_sorted, [r_sorted[N]])

# ╔═╡ 7aa3e93e-d7f7-41b9-aaeb-0a6f13eca4f5
md"🐜 use the `linear_interpolation` function to construct the function $r(\theta)$ (the linear interpolator of the data $\{(\theta_i, r_i)\}$) and assign it as a variable `r_of_θ`. pass the keyword argument `extrapolation_bc=Periodic()` to the function `linear_interpolation` for the interpolator to implement periodic boundary conditions.
"

# ╔═╡ 4d7ed267-7483-4fbb-9e4c-75b9474bb651
r_of_θ = linear_interpolation(θ_sorted, r_sorted; extrapolation_bc=Periodic())

# ╔═╡ 74e6ca9d-cb85-43d2-885c-2ab5ddaabe6f
md"🐜 to show the interpolator respects periodic boundary conditions, compute $r(\phi)$ and $r(\phi + 2 \pi)$ for some random $\phi$. they should match."

# ╔═╡ 3f2858b0-daec-4e61-9d8d-1ca7bc704428
begin
	ϕ = π/4
	r_of_θ(ϕ) == r_of_θ(ϕ + 2π)
end

# ╔═╡ b51e69d6-e80e-40b5-b835-59665b6cc9c5
md"🐜 to ensure your function $r(\theta)$ representing the continuous boundary of the lake makes sense, create a `range` of 500 $\theta$'s between $-\pi$ and $\pi$, compute the corresponding $r$'s, and plot the continuous-looking curve representing the boundary of the lake. also plot the (centered) data points, to ensure the parametric curve $r(\theta)$ is passing through the data points.
"

# ╔═╡ 54f19024-3808-40c0-91a2-ad36d6a99f9e
begin
	θ_values = range(-π, π, 500)
	r_values = r_of_θ.(θ_values)
	x = [r_of_θ(θ)*cos(θ) for θ in θ_values]
	y = [r_of_θ(θ)*sin(θ) for θ in θ_values]
	
	drawlake(X̂, (x, y))
end

# ╔═╡ 78bd4975-8098-44d6-91ec-977540a11644
md"## Monte Carlo
now that we have a continuous representation of the lake boundary via $r=r(\theta)$, we can run a Monte Carlo simulation to estimate the area of the lake.

🐜 write a function `inside_lake(x, r_of_θ)` that takes as arguments:
* a centered coordinate `x` (a vector)
* your interpolation-based representation of the lake boundary `r_of_θ`
and returns
* `true` if `x` falls inside the lake boundary
* `false` if `x` falls outside the lake boundary

!!! hint
	inside the function, compute the polar coordinates of `x`. compare with the output of `r_of_θ` giving the lake boundary.
"

# ╔═╡ 643006b0-d36e-4c7d-8179-5e52fa3366e1
function inside_lake((x, y), r_of_θ)
	r = √(x^2 + y^2)
	θ = atan(y, x)

	r_lake = r_of_θ(θ)

	return r <= r_lake
end

# ╔═╡ aa4fa8e2-f34f-43ab-859d-21322c15e6d2
md"🐜 test your function.
* `inside_lake([0.0, 0.0], r_of_θ)` should return `true`.
* `inside_lake([-2, 2.0], r_of_θ)` should return `false`.
"

# ╔═╡ 05b591a7-b70b-473c-9ad1-706e5bf95c0c
inside_lake([0.0, 0.0], r_of_θ)

# ╔═╡ 24ce43ec-89ae-4175-9837-c0ddffdee98f
inside_lake([-2, 2.0], r_of_θ)

# ╔═╡ 1ed3b8a8-34c4-44e5-876d-cecc4806a122
md"🐜 (conceptually) draw a square $[-L/2,L/2]^2$ around the lake boundary, with $L$ sufficiently large to include all of the lake.
assign the variable `L` as the maximum absolute value of a (centered) coordinate of the lake, to ensure the square encompasses the lake.
"

# ╔═╡ 6d65a99e-455b-4fa0-be1a-0ef77e368913
L = round(maximum(r_sorted) * 2; digits=2) # max r * 2

# ╔═╡ 59e6635c-f01f-4fc3-bb5a-873e2a88ff38
md"
think of a large, homogeneous raincloud covering the square of area $L^2$. the raincloud drops rain drops at uniform random locations in the square. we will simulate this by dropping raindrops at uniform random locations in the square. these raindrops can fall inside the lake or outside of it. the area of the lake is then $L^2$ times the fraction of the raindrops that fall inside the lake. 

🐜 write a function `run_monte_carlo(N, L, r_of_θ)` whose arguments are:
* `N`: the number of Monte Carlo samples (rain drops)
* `L`: the length of the square
* `r_of_θ`: the interpolation-based representation of $r(\theta)$ giving the lake boundary
and returns
* `X_rain`: a `2 × N` matrix whose columns give the locations where the raindrops fell
* `rain_inside`: a `N`-length boolean matrix indicating whether the raindrops fell inside or outside the lake.

!!! hint
	inside the function, you need a way to generate a uniform random number in $[-L/2, L/2]$. I suggest you use `Uniform` from the `Distributions.jl` package [here](https://juliastats.org/Distributions.jl/latest/univariate/#Distributions.Uniform). you could use `rand()` as well, with some extra steps.
"

# ╔═╡ 15d303b5-61c5-450e-8301-d7d2f0e822eb
function run_monte_carlo(N, L, r_of_θ)
	distr = Uniform(-L/2, L/2)
	X_rain = rand(distr, (2, N))
	rain_inside = [inside_lake(X_rain[:, i], r_of_θ) for i=1:N]

	return X_rain, rain_inside
end

# ╔═╡ de00d71a-52c0-4879-9b2d-fc9f8b5e3f94
md"🐜 run your Monte carlo simulation with `N=10000`.
compute your estimate for the area of the lake. it should be close to what Wikipedia reports, 20.6 mi² (not clear if this includes Wizard Island or not).

!!! note 
	if your area estimate does not match that of Wikipedia within 1 mi², there must be a bug. the next step will help you diagnose.
"

# ╔═╡ c1c5e37e-6a18-438e-ab56-376ecfecf3e7
function calc_area(N, L, r_of_θ)
	X_rain, rain_inside = run_monte_carlo(N, L, r_of_θ)
	n_inside = sum(rain_inside)
	area = L^2 * n_inside/N
	df = DataFrame(
		"x" => X_rain[1, :],
		"y" => X_rain[2, :],
		"rain_inside" => rain_inside
	)
	area, df
end

# ╔═╡ a7659496-6694-4134-952c-152e3f098ba5
area, df = calc_area(10_000, L, r_of_θ)

# ╔═╡ bab8f895-0913-49fb-ab6b-e49f0ab9af6a
area

# ╔═╡ 8781b9bc-25e3-471f-a478-877223186f4b
md"🐜 draw a plot containing:
* the centered coordinates of the lake (as above)
* the continuous interpolation-based representation of $r(\theta)$ as a curve (as above)
and, in addition, to visualize the result of your Monte Carlo simulation:
* each rain drop that fell in the square, with raindrops that fell in the lake colored blue and raindrops that fell outside the lake colored brown. if your code is correct, you should see the blue lake surrounded by brown. use the `:+` marker for the raindrops.
"

# ╔═╡ 12717627-4edc-445f-84df-5c3cc4c99a1c
outside_df, inside_df = groupby(df, "rain_inside")

# ╔═╡ 9a261341-e380-41d3-8845-cc5ee5608248
begin
	f = drawlake(X̂, (x, y))
	scatter!(outside_df[:, "x"], outside_df[:, "y"]; color = :brown, marker = :+)
	scatter!(inside_df[:, "x"], inside_df[:, "y"]; color = :blue, marker = :+)
	f
end

# ╔═╡ 27301229-b35e-4fb0-8bf9-c6fe43434821
md"
## written questions
* run the simulation a few times and qualitatively note the variance in the estimated area of the lake. now set `N=100` and do the same. explain the difference in variance between the estimates. what do you conclude?
* check out the map of Crater Lake. it includes Wizard Island. how would you go about accounting for the area of water only, excluding Wizard Island, as a next step?
* would this strategy to parameterize the lake as $r=r(\theta)$ work with all lakes? for example, look up the shape of Lake Huron. how would you write an `inside_lake` function that works for Lake Huron? (I'm not sure how I would approach this, myself...)
"

# ╔═╡ 7627b4d4-4663-468e-b5af-41d1ca53f96f
area2, _ = calc_area(100, L, r_of_θ)

# ╔═╡ 6576b16d-472c-436c-8bc8-abe74aab46f0
area2

# ╔═╡ 3faec184-f8e2-4778-807d-9230d99db339
md"""
At a glance, the runs with 10,000 samples appear to have significantly less variance, around .6 square miles. The runs with 100 samples have a much higher variance, around 3 square miles. This leads to a conclusion that a higher sample size reduces variance.

Wizard Island's area can be estimated using the same methods as above, namely by parameterizing a function that can determine whether a particular point is within the island's boundary. We could then run the Monte Carlo simulation to determine which points fall within the lake boundary and which points fall within the Island boundary. We could then determine the area as before, but subtracting the points that fall within the Island boundary from the lake's area.

This strategy would break down with a lake that is highly non-convex (highly variable border) such that there would be multiple $r$ values for any given $\theta$. One potential solution would be to partition the lake into manageable regions and then aggregate the area results.
"""

# ╔═╡ Cell order:
# ╠═640a599e-3a27-11ed-069c-f14f3b6633e9
# ╠═78a8749f-c501-4e7b-bccf-3bdf4d55df92
# ╟─dbf497fa-f45b-4f75-96c0-3e0a73af0203
# ╟─2e87f1a8-b943-4317-99b1-8125e99355f2
# ╠═4f83121f-4b98-4854-adb4-7e11be25b482
# ╟─60e98df1-e21a-4c8d-ba54-50455b61379c
# ╠═68ad49c5-6bd5-460c-b404-33e52292dd6d
# ╠═e9dbfd3f-a0d9-4dec-899b-aa62be5790dd
# ╠═010f8cc2-dbc9-4fa0-bc0a-2c088bfc00d5
# ╠═fd1a78dc-2fb4-48f9-a61c-fa00ecfd367b
# ╟─4ec36505-66d9-4cd0-a4c5-e3718d1a5cba
# ╠═b45e7b0a-8644-4210-bec9-8fd3076b08f4
# ╟─3a088002-b238-46eb-86f8-483ca5b9a7c2
# ╠═bc825fb1-5ee5-4d39-9ec2-a99ca8b68e79
# ╟─89dde6a1-1230-4841-8fca-c2563a54fd4a
# ╠═a9cf8d6e-1607-44e6-8f74-769ce7bff060
# ╟─9a6aa8a8-fd6a-4928-bf28-b7444b2e6d3b
# ╟─e5079f42-2137-459f-a351-ddc22864085a
# ╠═0edc2e54-6552-4a56-aa03-87ae8886e826
# ╟─d866aace-c019-4fe3-a77e-d681e1af001f
# ╠═6d01e662-761f-4f33-89d9-7d29c4196366
# ╟─cbce6c42-b5db-4b59-8399-495289c3b247
# ╠═d3a6961d-3f70-4b9e-8bc9-38e8ad91c952
# ╟─d80c35e7-ee90-464e-b7b4-6e2250efc3ff
# ╟─c11fa2f9-a440-4b0e-85ba-27c749f4fccb
# ╠═b4addc63-8f97-4c15-8e86-e0f5733071ef
# ╠═95d4cfa4-964d-45f0-8496-6a7beb382072
# ╠═6fe5a3de-ddda-426a-a072-1db6892f22aa
# ╟─d4a35029-ff51-41f4-ab80-e9e61455d0c1
# ╠═d9849427-3389-4040-900b-d4eadc922c7d
# ╠═4211a63e-c894-45fc-bb38-b13850dcae4d
# ╠═ef735579-310e-4c2d-91f0-f7ac99d71cf0
# ╟─7aa3e93e-d7f7-41b9-aaeb-0a6f13eca4f5
# ╠═4d7ed267-7483-4fbb-9e4c-75b9474bb651
# ╟─74e6ca9d-cb85-43d2-885c-2ab5ddaabe6f
# ╠═3f2858b0-daec-4e61-9d8d-1ca7bc704428
# ╟─b51e69d6-e80e-40b5-b835-59665b6cc9c5
# ╠═54f19024-3808-40c0-91a2-ad36d6a99f9e
# ╟─78bd4975-8098-44d6-91ec-977540a11644
# ╠═643006b0-d36e-4c7d-8179-5e52fa3366e1
# ╟─aa4fa8e2-f34f-43ab-859d-21322c15e6d2
# ╠═05b591a7-b70b-473c-9ad1-706e5bf95c0c
# ╠═24ce43ec-89ae-4175-9837-c0ddffdee98f
# ╟─1ed3b8a8-34c4-44e5-876d-cecc4806a122
# ╠═6d65a99e-455b-4fa0-be1a-0ef77e368913
# ╟─59e6635c-f01f-4fc3-bb5a-873e2a88ff38
# ╠═15d303b5-61c5-450e-8301-d7d2f0e822eb
# ╟─de00d71a-52c0-4879-9b2d-fc9f8b5e3f94
# ╠═c1c5e37e-6a18-438e-ab56-376ecfecf3e7
# ╠═a7659496-6694-4134-952c-152e3f098ba5
# ╠═bab8f895-0913-49fb-ab6b-e49f0ab9af6a
# ╟─8781b9bc-25e3-471f-a478-877223186f4b
# ╠═12717627-4edc-445f-84df-5c3cc4c99a1c
# ╠═9a261341-e380-41d3-8845-cc5ee5608248
# ╟─27301229-b35e-4fb0-8bf9-c6fe43434821
# ╠═7627b4d4-4663-468e-b5af-41d1ca53f96f
# ╠═6576b16d-472c-436c-8bc8-abe74aab46f0
# ╟─3faec184-f8e2-4778-807d-9230d99db339
