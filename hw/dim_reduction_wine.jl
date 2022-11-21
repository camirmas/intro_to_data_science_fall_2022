### A Pluto.jl notebook ###
# v0.19.12

using Markdown
using InteractiveUtils

# ╔═╡ e35f94a9-fd14-4617-ae89-6033d821a9c0
begin
	using Pkg

	Pkg.activate(Base.current_project())
	
	using CSV, DataFrames, ScikitLearn, Statistics, CairoMakie

	import AlgebraOfGraphics as aog
end

# ╔═╡ 5e1dee9b-b6d5-414d-a890-e756553bd16f
aog.set_aog_theme!(); update_theme!(fontsize=16)

# ╔═╡ b14100df-7d6b-4d9f-8b70-b91e4b382be0
@sk_import decomposition: PCA

# ╔═╡ e7701be9-a8eb-4b44-9875-9e61d35a154d
md"# principal component analysis (PCA) of wines


_source_: UCI Machine Learning repository [here](https://archive.ics.uci.edu/ml/datasets/wine).

> these data are the results of a chemical analysis of wines grown in the same region in Italy but derived from three different cultivars. The analysis determined the quantities of 13 constituents found in each of the three types of wines.


!!! warning
	ignore the variety of the wine in the first column---the _labels_ of the wine. we are in the territory of unsupervised learning, where we only have attributes of the instances (wines) but no labels (variety). we will conduct PCA on the wines using their attributes only, then assess if the structure of the data set (the scatter of the wine attributes represented as vectors in the feature space) captures information about the variety of the wines.

🍷 read in the wine data as a `DataFrame`.
"

# ╔═╡ 03981f54-8d9b-4246-99e9-9f33c818266c
header = ["Variety", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium", "Total phenols", "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"]

# ╔═╡ a94e462b-96d3-4b7f-b5e8-1e84eca5b7a5
begin
	url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
	download(url, "wine_data.csv")
	wine_data = CSV.read("wine_data.csv", DataFrame; header=header)
end

# ╔═╡ dd39776a-b853-4617-ada9-16a50ff2044a
md"🍷 construct the (# wines) × (# attributes) feature matrix `X` that lists the attributes of the wines in the rows. so each row represents a wine, and each column represents an attribute of the wines. be sure not to include the `\"Variety\"`, as this is a label.

!!! hint
	`Matrix(data[:, [\"col x\", \"col y\"]])` will grab two columns from a data frame and convert it to a matrix.
"

# ╔═╡ 154586dd-ae21-4c1a-80a9-7a045557a15f
X = Matrix(select(wine_data, Not("Variety")))

# ╔═╡ 16b0364d-6172-4ec6-977f-7173ada77e57
y = select(wine_data, "Variety")

# ╔═╡ f87ffcb2-c68a-4efa-8bc4-881e8a53aab4
md"🍷 PCA is most effective when the values of the features are standardized. loop through each column of the feature matrix `X` and standardize each feature by (i) subtracting the mean value of that feature among the instances and (ii) dividing by the standard deviation of the values of the feature among the instances. you should notice that each value of the feature tends to lie in $[-2, 2]$, but outliers can lie outside of the interval.
"

# ╔═╡ d1831d6d-93bb-4e8a-a731-8ad3650336d3
function standardize(X)
	X_new = copy(X)

	n_rows, n_cols = size(X)

	for col_i in 1:n_cols
		col = X_new[:, col_i]
		avg = mean(col)
		stdev = std(col)
		X_new[:, col_i] = (col .- avg) ./ stdev 
	end

	X_new
end

# ╔═╡ a0060432-6228-4704-b971-13010afa152c
standardize(X)

# ╔═╡ 24b78de1-d624-4e02-97c3-b4fb55234e46
md"🍷 use `PCA` from `scikit-learn` and its `fit_transform` method to conduct PCA on the wine data. particularly, embed each wine, originally represented as a 13-dimensional (the # of attributes) feature vector, into a 2D space conducive for visualization. i.e. retain only the first two principal components."

# ╔═╡ c56868a3-a4e1-4e03-aba7-538f4b725b02
begin
	n_rows, n_cols = size(X)
	pca = PCA(n_components=n_cols)
	Xₛ = standardize(X)
	Xₜ = pca.fit_transform(Xₛ)[:, 1:2]
end

# ╔═╡ 2f8c95de-5b38-444f-8101-0bdfe12221e1
pca.components_

# ╔═╡ 4727f313-b81a-4939-bb8d-5ffa2d9100c4
md"🍷 visualize the first two principal components of each wine. i.e. plot the 2D embeddings of the wines. color each point (representing a wine) by the variety of wine it belongs to, the labels in the first column of the wine data that we held-out from the unsupervised PCA. include a legend to indicate which color corresponds to which wine variety (1, 2, 3).
"

# ╔═╡ 1b4a9e8f-ea94-45d9-8559-65084c688f14
begin
	fig = Figure()
	ax = Axis(fig[1, 1], xlabel="PC1", ylabel="PC2")
	pc1 = Xₜ[:, 1]
	pc2 = Xₜ[:, 2]

	varieties = y[:, 1]
	n_varieties = varieties |> unique |> length
	colors = [:red, :blue, :green]
	plots = Vector(undef, 3)
	
	for var_i in unique(varieties)
		ids = [i for (i, var) in enumerate(varieties) if var == var_i]
		pc1_var = pc1[ids]
		pc2_var = pc2[ids]
		plt = scatter!(pc1_var, pc2_var, color=colors[var_i], label="Variety $var_i")
		plots[var_i] = plt
	end

	leg = Legend(fig[1, 2], ax)

	fig
end

# ╔═╡ a12141d9-087d-4306-a822-dd96ed83d416
md"""
🍷judging from your plot, was PCA able to find meaningful patterns/structure in the 13-dimensional wine feature vectors, even though we retained only two dimensions?

There does appear to be a meaningful pattern when we observe the 13 dimensional wine features projected onto only the first two principal components, as the different varieties appear to form 3 distinct groups. This may be useful moving forward in to a classification problem, as we know that we have captured enough variance with these two PCs to identify distinct groups.
"""

# ╔═╡ 043c47e5-1d82-4d4b-8740-85cffe445cad
md"🍷 what percentage of the variance among the 13-dimenionsal feature vectors were the first two principal components able to, together, capture? see the `explained_variance_ratio_` attribute of your fitted PCA model."

# ╔═╡ cd82e4e4-f38a-4718-9ad3-1a3a6d5c7212
sum(pca.explained_variance_ratio_[1:2])

# ╔═╡ Cell order:
# ╠═e35f94a9-fd14-4617-ae89-6033d821a9c0
# ╠═5e1dee9b-b6d5-414d-a890-e756553bd16f
# ╠═b14100df-7d6b-4d9f-8b70-b91e4b382be0
# ╟─e7701be9-a8eb-4b44-9875-9e61d35a154d
# ╠═03981f54-8d9b-4246-99e9-9f33c818266c
# ╠═a94e462b-96d3-4b7f-b5e8-1e84eca5b7a5
# ╟─dd39776a-b853-4617-ada9-16a50ff2044a
# ╠═154586dd-ae21-4c1a-80a9-7a045557a15f
# ╠═16b0364d-6172-4ec6-977f-7173ada77e57
# ╟─f87ffcb2-c68a-4efa-8bc4-881e8a53aab4
# ╠═d1831d6d-93bb-4e8a-a731-8ad3650336d3
# ╠═a0060432-6228-4704-b971-13010afa152c
# ╟─24b78de1-d624-4e02-97c3-b4fb55234e46
# ╠═c56868a3-a4e1-4e03-aba7-538f4b725b02
# ╠═2f8c95de-5b38-444f-8101-0bdfe12221e1
# ╟─4727f313-b81a-4939-bb8d-5ffa2d9100c4
# ╠═1b4a9e8f-ea94-45d9-8559-65084c688f14
# ╟─a12141d9-087d-4306-a822-dd96ed83d416
# ╟─043c47e5-1d82-4d4b-8740-85cffe445cad
# ╠═cd82e4e4-f38a-4718-9ad3-1a3a6d5c7212
