### A Pluto.jl notebook ###
# v0.19.12

using Markdown
using InteractiveUtils

# ╔═╡ d566a523-323b-4001-871a-1956031fd289
begin

	using Pkg
	Pkg.activate(Base.current_project())
	using PlutoUI, CSV, DataFrames, CairoMakie
	using AlgebraOfGraphics
	import ScikitLearn
end

# ╔═╡ 26c3ac2a-e83c-445e-b841-258802724124
set_aog_theme!()

# ╔═╡ 583b698d-79ab-410f-aaeb-a09c471c83dd
TableOfContents()

# ╔═╡ f9143065-f790-4b22-bd30-0ebf8b1b8a8f
begin
	ScikitLearn.@sk_import ensemble : RandomForestClassifier
	ScikitLearn.@sk_import metrics : confusion_matrix
end

# ╔═╡ dc73c588-586f-11ed-334d-a5e71913f232
md"# mushroom classification via random forests

!!! note \"objective\"
	train and evaluate a random forest to classify mushrooms as poisonous or not, based on their attributes.

## the labeled data

🍄 download and read in the mushroom data set from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/mushroom).

!!! hint
	see the `.names` file, section 7, for the attributes. see the `header` kwarg of `CSV.read` to give the appropriate column names.
"

# ╔═╡ 2d27a20a-6699-4b18-b465-87a6080c62d8
header = ["class", "cap-shape", "cap-surface", "cap-color", "bruises?", 
	      "odor", "gill-attachment", "gill-spacing", "gill-size", 
	      "gill-color", "stalk-shape", "stalk-root", "stalk-surface-above-ring",
	      "stalk-surface-below-ring", "stalk-color-above-ring", 
	      "stalk-color-below-ring", "veil-type", "veil-color", "ring-number",
		  "ring-type", "spore-print-color", "population", "habitat"] # you're welcome

# ╔═╡ 9702bafd-6847-49bb-bba9-6524fafea360
begin
	url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
	download(url, "mushroom_data.csv")
	mushrooms = CSV.read("mushroom_data.csv", DataFrame, header=header)

	url_names = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.names"
	download(url_names, "mushroom_names.txt")

	mushrooms
end

# ╔═╡ 9d47ccb0-a4c5-4bfe-8856-890850f2f13d
function get_mushroom_data()
	mushrooms = CSV.read("mushroom_data.csv", DataFrame, header=header)

	return mushrooms
end

# ╔═╡ ecf6d038-f102-4dfb-b75b-559977e19876
md"🍄 how many features (attributes) of the mushrooms are recorded in the data set?"

# ╔═╡ 253cb90c-702c-4fa4-84ed-4c390bfbcc3f
ncol(mushrooms)

# ╔═╡ 1334afa0-692a-4818-9444-bce9971a5415
md"🍄 are there any missing values in the data? if so, drop the rows that contain `missing` entries."

# ╔═╡ 8abf9193-f518-49df-8e0e-16f8e0b89d9a
begin
	mushrooms_missing = CSV.read("mushroom_data.csv", DataFrame; missingstring="?", header=header)
	dropmissing(mushrooms_missing)
end

# ╔═╡ 48cb6573-d126-4571-8846-fc15f0ea801b
md"🍄 use `combine` and `groupby` to determine how many of the mushrooms are edible vs. not."

# ╔═╡ 98a2d321-06f6-46b8-9f05-339eeaf00401
combine(groupby(mushrooms, :class), nrow => "Count")

# ╔═╡ d983df60-8957-4ba4-a981-e8797851da09
md"🍄 the features are categorical. how many unique categories does each feature have?

!!! hint
	I used `combine` with `All() .=> ...`.
"

# ╔═╡ fea46872-5758-4b65-8ab8-362c56cedcb1
combine(mushrooms, All() .=> col -> length(unique(col)); renamecols=false)

# ╔═╡ fbd0b2ff-2ed4-40d6-ad7a-b89b43208056
md"## exploring the data

🍄 my hypothesis is that odor alone can be used to distinguish between edible and poisonous mushrooms with reasonable accuracy. to test this hypothesis, draw a bar plot such that:
* each possible odor is listed on the x-axis
* there are two bars side-by-side for each odor: one representing poisonous mushrooms, the other representing edible mushrooms
* the height of the bar represents the number of mushrooms with that class label _and_ that odor
* the bars are colored differently according to class
* a legend indicates which color corresponds to which class
* the class name and odors are spelled out to be legible. e.g. instead of \"e\" we have \"edible\" in the legend; instead of \"n\" we have \"none\" as the odor label on the x-axis.

!!! hint
	you can do this manually via a double `groupby` and the `dodge` kwarg of `barplot`. however, I found it much easier to use `AlgebraOfGraphics.jl`, which shows an analogous example [here](https://aog.makie.org/stable/generated/penguins/#Styling-by-categorical-variables).

"

# ╔═╡ 491a16bf-599a-4370-a2e1-17284ae96e8e
mushroom_frequency = data(mushrooms) * frequency() * mapping(:odor; dodge=:class, color=:class)

# ╔═╡ a6abd64b-b826-430f-ae90-58a51dfd4e5d
begin
	axis = (; ylabel="Mushroom Count")
	f = draw(mushroom_frequency; axis=axis)
	ylims!(0, nothing)
	
	f
end

# ╔═╡ b273e7ee-625f-4da4-88bf-fa6e2cb3c912
md"## preparing the features for machine learning

❗ the `RandomForestClassifier` in scikit-learn does not a categorical variable as an input if the variable has more than two categories. for example, there are nine unique categories of odors. the random forest algorithm implementation cannot handle this. however, the algorithm _can_ handle binary features. so, we will convert each multi-category feature into a set of binary feature. for example, for the odor feature, we convert it into nine different binary indicator variables.

_old feature_: odor (values it can take on: pungent, almond, anise, none, foul, creosote, fishy, spicy, musty)

_new features encoding the same information_:
* odor_pungent (values it can take on: 0, 1)
* odor_almond (values it can take on: 0, 1)
...
* odor_musty (values it can take on: 0, 1)

🍄 create a new set of 117 binary features, as new columns in the mushrooms data frame, that encode the same attributes about the mushrooms as the original features. name these columns appropriately so that we can understand what each column means. for example, \"odor=pungent\" should be the column name of one of the new binary features.

!!! hint
	I used a double `for` loop and the `transform` function.
"

# ╔═╡ 18467ce4-244c-4290-9d71-3172e30aef4c
function convert_features!(mushrooms)
	colnames = names(select(mushrooms, Not(:class)))

	for colname in colnames
		col = mushrooms[:, colname]
		features = col |> unique
		total_cols = features |> length
		total_rows = nrow(mushrooms)

		for feat in features	
			name = "$colname=$feat"
			values = zeros(Int, total_rows)
			mushrooms[:, name] = Int.([val == feat for val in col])
		end
	end

	mushrooms
end

# ╔═╡ bde95c72-8442-4dca-9e3f-76768d6ae053
begin
	local mushrooms = get_mushroom_data()
	convert_features!(mushrooms)
end

# ╔═╡ 8fc789e8-3f31-46f7-932d-ae3489c8132a
md"🍄 create the (# mushrooms × # binary features) feature matrix `X` with the binarized feature vector of each mushroom in the rows.
"

# ╔═╡ 0b6dbd95-2e42-4235-9da1-cdb93e0e7eab
begin
	local mushrooms = get_mushroom_data()
	convert_features!(mushrooms)
	X = Matrix(select(mushrooms, Not(header)))
	y = mushrooms[:, "class"]
	X
end

# ╔═╡ ae66dc33-9e0b-4b3a-a3ba-65d338e7add3
md"🍄 create a # mushrooms-length target vector `y` listing the labels of the mushrooms. of course, the rows of `y` and rows of `X` must refer to the same mushroom..."

# ╔═╡ dd27778b-7385-417a-a6d6-07d087a78f85
y # see above

# ╔═╡ 54b4c02f-3c61-44f4-97d3-3e4dd6b2aea5
md"## train and evaluate the random forest classifier

🍄 grow a random forest classifier using _all_ of the data. 

!!! note
	what are the labeled input-output pairs for the random forest here? the input is the vector of binarized features representing attributes of the mushrooms. the output is the label of poisonous or edible.

ensure the random forest is set up to evaluate the predictions on the out-of-bag samples to justify _not_ using a train/test split. use the default settings of the random forest, which tend to work well out-of-the-box.

!!! hint
	see the scikit-learn docs for `RandomForestClassifier` [here](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier.feature_importances_). use the `fit` method, as you did with decision trees.
"

# ╔═╡ eca30527-400d-4a70-a66a-84599b174130
rfc = RandomForestClassifier(oob_score=true)

# ╔═╡ be517817-fa78-47f1-a8b6-e7bcf13edb37
rfc.fit(X, y)

# ╔═╡ 0c51f438-0bd4-4aef-aec8-b3a5ec437f41
md"🍄 how many decision trees comprise your random forest?"

# ╔═╡ 3417f6dd-4818-4934-9a0a-b43d5d986c8a
md"The default number of trees is 100."

# ╔═╡ c4e01a74-d669-40e2-9510-0435b53fff60
md"🍄 draw a histogram of the depths of the trees in the forest. you should see a diversity of depths, owing to the randomness of the trees.

!!! hint
	see the `estimators_` attribute, which gives a list of the individual decision trees. then like the previous assignment, use `get_depth()`.
"


# ╔═╡ 4cdd257a-68d4-4a44-aabe-258cccb202a1
begin
	num_trees = rfc.estimators_ |> length
	depths = [tree.get_depth() for tree in rfc.estimators_]
end

# ╔═╡ 8e3f2699-0e0b-41b5-a429-12256e91ba24
begin
	f2 = Figure()
	ax = Axis(f2[1, 1], xlabel="Tree Depth", ylabel="Count")
	hist!(depths)

	f2
end

# ╔═╡ d66d25ca-07c4-490e-a2ae-64fb49ab634d
md"🍄 write the two major ways by which random forests inject randomness into each decision tree that comprises it. this leads to the decorrelated trees.

1. each tree is trained on a random subset of the training set
2. at each split within a tree, a random subset of the predictors are available
"

# ╔═╡ dc2ac948-c629-4223-8005-7793ad8ad30d
md"🍄 compute the confusion matrix using the out of bag predictions on the mushrooms.

!!! hint
	see the `oob_decision_function_` attribute of the random forest classifier.
"

# ╔═╡ 09766213-b16d-42ae-8852-96b4f6ceca2d
y_pred = rfc.predict(X)

# ╔═╡ 5ba7af92-b2a7-48f7-b2de-73c74900f364
oob = rfc.oob_decision_function_

# ╔═╡ caaeb39a-3ada-433e-b410-e1f1bc5f71bb
rfc.classes_

# ╔═╡ 1146dcc7-53ce-430d-8675-ac232903f418
confusion_matrix(y, y_pred)

# ╔═╡ d7cbd4d9-5247-4255-9091-a4e6fb473737
rfc.oob_score_

# ╔═╡ 9e5bca95-c027-447c-82f9-a21499df43d9
md"🍄 precisely explain what the \"out of bag prediction\" for a given mushroom means.

in the out-of-bag prediction for a given mushroom, we...
"

# ╔═╡ 36ab9afe-611d-464d-87f8-ba91b791fc1b
md"
## feature importance
random forests are almost always more accurate than an individual decision tree, and they are much easier to train because little tuning is needed. however, we lose interpretability because the decision of which label to place on a mushroom is being made by a committee of trees instead of just one.

🍄 compute the impurity-based importance of each feature, which is kept track of while growing the tree. draw a bar plot that shows, _for the ten most important features_:
* y-axis: the ten most important features
* x-axis: the importance score of those features
* so, bar lengths = the importance score

!!! hint
	see the `feature_importances_` attribute of the random forest classifier.
" 

# ╔═╡ 490ce163-afe6-4ed1-8a4e-0ae6a6f86a15
begin
	local mushrooms = get_mushroom_data()
	convert_features!(mushrooms)
	mushrooms = select(mushrooms, Not(header))
	
	local sort_ids = sortperm(rfc.feature_importances_; rev=true)[1:10]
	local y = rfc.feature_importances_[sort_ids]
	local col_names = names(mushrooms)[sort_ids]
	
	local f = Figure()
	local ax = Axis(
		f[1, 1]; 
		yticks=(1:10, col_names), 
		xlabel="Feature Importance", 
		ylabel="Feature"
	)
	barplot!(1:10, y; direction=:x)
	xlims!(0, nothing)

	f
end

# ╔═╡ Cell order:
# ╠═d566a523-323b-4001-871a-1956031fd289
# ╠═26c3ac2a-e83c-445e-b841-258802724124
# ╠═583b698d-79ab-410f-aaeb-a09c471c83dd
# ╠═f9143065-f790-4b22-bd30-0ebf8b1b8a8f
# ╟─dc73c588-586f-11ed-334d-a5e71913f232
# ╠═2d27a20a-6699-4b18-b465-87a6080c62d8
# ╠═9702bafd-6847-49bb-bba9-6524fafea360
# ╠═9d47ccb0-a4c5-4bfe-8856-890850f2f13d
# ╟─ecf6d038-f102-4dfb-b75b-559977e19876
# ╠═253cb90c-702c-4fa4-84ed-4c390bfbcc3f
# ╟─1334afa0-692a-4818-9444-bce9971a5415
# ╠═8abf9193-f518-49df-8e0e-16f8e0b89d9a
# ╟─48cb6573-d126-4571-8846-fc15f0ea801b
# ╠═98a2d321-06f6-46b8-9f05-339eeaf00401
# ╟─d983df60-8957-4ba4-a981-e8797851da09
# ╠═fea46872-5758-4b65-8ab8-362c56cedcb1
# ╟─fbd0b2ff-2ed4-40d6-ad7a-b89b43208056
# ╠═491a16bf-599a-4370-a2e1-17284ae96e8e
# ╠═a6abd64b-b826-430f-ae90-58a51dfd4e5d
# ╟─b273e7ee-625f-4da4-88bf-fa6e2cb3c912
# ╠═18467ce4-244c-4290-9d71-3172e30aef4c
# ╠═bde95c72-8442-4dca-9e3f-76768d6ae053
# ╟─8fc789e8-3f31-46f7-932d-ae3489c8132a
# ╠═0b6dbd95-2e42-4235-9da1-cdb93e0e7eab
# ╟─ae66dc33-9e0b-4b3a-a3ba-65d338e7add3
# ╠═dd27778b-7385-417a-a6d6-07d087a78f85
# ╟─54b4c02f-3c61-44f4-97d3-3e4dd6b2aea5
# ╠═eca30527-400d-4a70-a66a-84599b174130
# ╠═be517817-fa78-47f1-a8b6-e7bcf13edb37
# ╟─0c51f438-0bd4-4aef-aec8-b3a5ec437f41
# ╟─3417f6dd-4818-4934-9a0a-b43d5d986c8a
# ╟─c4e01a74-d669-40e2-9510-0435b53fff60
# ╠═4cdd257a-68d4-4a44-aabe-258cccb202a1
# ╠═8e3f2699-0e0b-41b5-a429-12256e91ba24
# ╟─d66d25ca-07c4-490e-a2ae-64fb49ab634d
# ╟─dc2ac948-c629-4223-8005-7793ad8ad30d
# ╠═09766213-b16d-42ae-8852-96b4f6ceca2d
# ╠═5ba7af92-b2a7-48f7-b2de-73c74900f364
# ╠═caaeb39a-3ada-433e-b410-e1f1bc5f71bb
# ╠═1146dcc7-53ce-430d-8675-ac232903f418
# ╠═d7cbd4d9-5247-4255-9091-a4e6fb473737
# ╟─9e5bca95-c027-447c-82f9-a21499df43d9
# ╟─36ab9afe-611d-464d-87f8-ba91b791fc1b
# ╠═490ce163-afe6-4ed1-8a4e-0ae6a6f86a15
