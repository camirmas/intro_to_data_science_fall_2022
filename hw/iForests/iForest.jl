### A Pluto.jl notebook ###
# v0.19.12

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 5c87a3f8-1f74-4918-94c5-3ba9764d6cde
begin
	import Pkg
	Pkg.activate(Base.current_project())

	using CSV, ScikitLearn, DataFrames, CairoMakie, AlgebraOfGraphics, PlutoUI, ColorSchemes
end

# ╔═╡ 87ebcca3-7e0a-476f-9b05-1893af8750d2
md"# isolation forests for anomaly detection

!!! example \"the task\"
	detect anomalous samples of glass using an isolation forest.

for background reading on isolation forests, see the original paper of Liu, Ting, and Zhou [here](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf?q=isolation-forest).

"

# ╔═╡ 6a6bea6d-e6e4-4401-8566-df0dc96a6855
TableOfContents()

# ╔═╡ 001320a7-1078-4a79-8b49-cbf9222dcc3d
set_aog_theme!(); update_theme!(resolution=(500, 500), fontsize=18)

# ╔═╡ c2451b7d-f1a9-4d66-a7f1-1f3176d04621
begin
	@sk_import ensemble : IsolationForest 
	@sk_import metrics : confusion_matrix
end

# ╔═╡ 8a835846-2497-11eb-167d-d98f197b3895
md"
## read in, process the glass data

_data source_: UCI machine learning repository [here](https://archive.ics.uci.edu/ml/datasets/glass+identification).

each sample is a piece of glass. for each glass sample, we have measurements of the sodium (Na) and silicon (Si) content (units: weight %). these are the two features. each sample is labeled as normal (glass coming from building windows) or anomalous (glass coming from tableware).

!!! warning \"conceptually, ignore the labels!\"
	in real-life anomaly detection settings, we do _not_ have the labels! we will not show these labels to the anomaly detection algorithm; rather, we have them here to test the performance of the anomaly detection algorithm. i.e. we will test whether the anomaly detection algorithm can correctly pick out the anomalous samples.

🐸 the data for this assignment is contained in the data frame `glass`.
"

# ╔═╡ 4a1d9e55-b0a0-4988-8173-9c33ce349430
begin
	# download the glass data from UCI repository
	download(
		"https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data",
		"glass.data"
	)
	
	# read in data as a DataFrame. header is from glass.names
	_glass = CSV.read(
		"glass.data", 
		header=["id", "refractive_index", "Na", "Mg", "Al", 
			    "Si", "K", "Ca", "Ba", "Fe", "class"], 
		DataFrame
	)
	
	# remove all samples except class 1 and 6.
	filter!(row -> row["class"] in [1, 6], _glass)
	
	# rename class 1 as normal, class 2 as anomalous
	transform!(
		_glass, 
		"class" => col -> map(c -> c == 1 ? "normal" : "anomaly", col),
		renamecols=false
	)

	# select only relevant columns
	rename!(_glass, "class" => "label")
	glass = select(_glass, ["Na", "Si", "label"])
	
	glass
end

# ╔═╡ 56fbbfb9-97d9-4f79-af2e-6d5b04c11be5
md"🐸 how many glass samples are there in the data set?"

# ╔═╡ 77657312-7aab-4e91-b0d9-e362a6305fdc
n_samples = glass |> nrow

# ╔═╡ 124801c2-eadb-444b-b882-0f847dd93754
md"🐸 what is the label distribution in the data set (i.e., how many normal vs. anomalous?)?
"

# ╔═╡ a37680d0-2557-11eb-3808-eb7d07f9c3f0
begin
	norm, anom = groupby(glass, "label")

	nrow(norm), nrow(anom)
end

# ╔═╡ 4b7bc817-adbf-456f-92be-c6b3218a850c
md"## viz the data"

# ╔═╡ 9ec36fe1-95ed-446b-b444-6a0331e64c1d
md"
🐸 visualize the scatter of the samples of glasses in the 2D feature space. controlled by the check box below, color each data point according to the label and add a legend if `color_by_label` is `true`. otherwise, just draw the data as black points, which reflects the anomaly detection setting where we do _not_ know the labels on the samples.

!!! note
	think: do the anomalous examples seems susceptible to isolation by an iForest?
"

# ╔═╡ dc5356b8-0649-4e1c-aa35-5c202ef41c36
md"color by label? $(@bind color_by_label PlutoUI.CheckBox())"

# ╔═╡ 71f35623-5522-4d26-a4fe-c225fb0d676a
colors = Dict("anomaly" => "red", "normal" => "green")

# ╔═╡ d3422028-5f9d-4518-a48e-19a690570fa1
function draw_glass(norm, anom)
	fig = Figure()
	ax = Axis(fig[1, 1], xlabel="Na", ylabel="Si")
	scatter!(Matrix(norm[:, 1:2]), color=colors["normal"], label="normal")
	scatter!(Matrix(anom[:, 1:2]), color=colors["anomaly"], label="anomaly")

	leg = Legend(fig[1, 2], ax)

	fig
end

# ╔═╡ bbb8abc0-457d-479c-aa8a-5883298950d3
draw_glass(norm, anom)

# ╔═╡ d5fa7759-10a8-4fd3-b15f-1abf69d19fc8
md"## training an iForest"

# ╔═╡ a4bc5d3f-20e5-4796-8c60-ac2abd7e5eb0
md"
🐸 to prepare for input to scikit-learn, construct a feature matrix. rows are samples. the two columns contain measurements of the two features of each sample."

# ╔═╡ 320c3c6d-4e8b-4197-b4af-06409e11f363
X = Matrix(select(glass, Not("label")))

# ╔═╡ bf635a00-7952-42ae-a855-3b7de1709c45
md"
🐸 grow an isolation forest (iForest) using all of the data. see the [scikit-learn docs on the iForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html).
"

# ╔═╡ 1c6b141a-af6b-4967-b571-6758da010790
iforest = IsolationForest()

# ╔═╡ 928c473d-119b-4341-9ce2-04d1d2e615de
iforest.fit(X)

# ╔═╡ 7ab8ad79-5a46-444a-ae19-70804f9a7ae2
md"
## the distribution of anomaly scores
🐸 compute anomaly scores from the iForest on all of the samples. visualize the distribution of anomaly scores. use the output of the `decision_function` method, so that an anomaly score of 0 is the boundary between anomalous and normal. see the [docs](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html). a negative score implies a prediction of anomalous; positive means normal. 
"

# ╔═╡ 8ac7e9bc-005e-442a-899a-7714b595152e
anomaly_scores = iforest.decision_function(X)

# ╔═╡ 75d6d25d-a7fc-42b1-a4bc-161018dc83ab
begin
	fig2 = Figure()
	ax2 = Axis(fig2[1, 1])

	barplot!(1:n_samples, anomaly_scores)

	ylims!(-0.5, 0.5)

	fig2
end

# ╔═╡ 6f70cbed-3386-4478-850d-0b791153c48e
md"
## making predictions for new samples

🐸 suppose two new glass samples were acquired.

* sample 1: Na wt % = 14.8, Si wt % = 73.1
* sample 2: Na wt % = 12.8, Si wt % = 73.15
construct their feature vectors below, the use the iForest to score them as anomalous or not. do the predictions by the iForest make sense, based on where these samples fall in feature space and the labels on the training data nearby it?
"

# ╔═╡ 1cb15395-d9b1-4198-84f6-3e26b3dfe294
begin
	samples = [
		14.8 73.1
		12.8 73.15
	]
	pred = iforest.predict(samples)
end

# ╔═╡ 50bc5514-0b6f-460c-babc-743fa7bba925
begin
	local fig = draw_glass(norm, anom)
	scatter!(samples', color=:blue, marker=:+, markersize=20)

	fig
end

# ╔═╡ 707178e8-cd43-4b3b-aaf3-aafd04a2f525
md"""
Looking at the data in comparison to the scatter plot, it is clear that the first sample falls well outside the typical values for Na but within the typical values for Si, which supports the prediction that it is an anomaly. The second sample appears to fall within the typical values for both features. 
"""

# ╔═╡ 654d3af3-f3e7-4f86-a022-653acc02ddb5
md"
## visualizing the anomaly detector

we have the luxury of doing this because our feature space only 2D.

🐸 interpret the iForest by visualizing: 
* the anomaly score of each point in feature space
* the decision boundary of the iForest anomaly detector (anomaly vs. normal)
* the scatter of the training samples, colored by class (❗ _which was unknown to the anomaly detector during training_!)
* employ a diverging colormap for the heat map and a colorbar to indicate correspondence. make sure the zero anomaly score corresponds to white (the middle of the colormap).

!!! hint
	lay a grid over feature space. compute the anomaly score by the iForest at each of these points and store it in a matrix. use `heatmap` to visualize the anomaly score and `contour` to visualize the decision boundary. 
"

# ╔═╡ e39a4aad-0d60-41fd-9304-6cf7c6043e47
cmap = reverse(ColorSchemes.diverging_gwr_55_95_c38_n256)

# ╔═╡ 00967516-a826-4cba-8222-4cf7d4404738
begin
	Na_min = minimum(X[:, 1])
	Na_max = maximum(X[:, 1])
	Si_min = minimum(X[:, 2])
	Si_max = maximum(X[:, 2])

	size = 25
	
	Na = range(Na_min, Na_max, size)
	Si = range(Si_min, Si_max, size)
	grid = zeros(size, size)
	
	for (i, Na_val) in enumerate(Na)
		for (j, Si_val) in enumerate(Si)
			grid[i, j] = iforest.decision_function([Na_val Si_val])[1]
		end
	end
	grid
end

# ╔═╡ 2939982e-4a78-4087-bbd9-30241c486efb
begin
	fig = Figure()
	ax = Axis(fig[1, 1], xlabel="Na", ylabel="Si")

	hm = heatmap!(Na, Si, grid, colormap=cmap, colorrange=(-.5, .5))
	contour!(Na, Si, grid, levels=[0], color=:green)
	Colorbar(fig[:, end+1], hm)
	
	fig
end

# ╔═╡ 1a8bc38d-f4f4-452f-81f3-ca3706c7f537
md"
## analyze the performance of the anomaly detector
🐸 compute and draw a confusion matrix showing the predicted vs. ground-truth labels. again, use the [decision_function method](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html#sklearn.ensemble.IsolationForest.decision_function).

!!! note
	since the labels were held out from the training process, this is a fair test of the anomaly detector.
"

# ╔═╡ ebadc73b-ac39-4012-9332-08d2d758d2f8
begin
	y_true = glass[:, "label"]
	label(x) = if x < 0 "anomaly" else "normal" end
	y_pred = map(label, iforest.decision_function(X))
	
	cm = confusion_matrix(y_true, y_pred)
end

# ╔═╡ dce3578c-3170-43fe-b448-b6ffe0ec4901
function viz_confusion_matrix(cm::Matrix{Int64}, class_list::Vector{String})
    fig = Figure()
    ax = Axis(fig[1, 1],
              xticks=([1, 2], class_list),
              yticks=([1, 2], class_list),
			  xticklabelrotation=45.0,
              ylabel="true",
              xlabel="prediction",
			aspect=DataAspect()
    )
    hm = heatmap!(cm, colormap=ColorSchemes.algae, colorrange=(0, sum(cm)))
    for i = 1:2
        for j = 1:2
            text!("$(cm[i, j])",
                  position=(i, j), align=(:center, :center), color=:black)
        end
    end
    Colorbar(fig[1, 2], hm, label="# glass samples")
    fig
end

# ╔═╡ 48f0a95d-f0e8-43aa-8f7c-651d9656e2af
viz_confusion_matrix(cm, ["anomaly", "normal"])

# ╔═╡ 7b5fdf78-c154-40fe-940b-477b49bb38f8
md"🐸 how many false alarms (thinking of an alarm as an anomaly)? how many anomalous samples passed through the iForest without alarm?"

# ╔═╡ 22ce06ce-2040-419d-996f-5fa13aab8516
md"""
There are 4 false alarms (predicted anomaly but was normal). There were 6 anomalous samples that passed without alarm.
"""

# ╔═╡ Cell order:
# ╟─87ebcca3-7e0a-476f-9b05-1893af8750d2
# ╠═5c87a3f8-1f74-4918-94c5-3ba9764d6cde
# ╠═6a6bea6d-e6e4-4401-8566-df0dc96a6855
# ╠═001320a7-1078-4a79-8b49-cbf9222dcc3d
# ╠═c2451b7d-f1a9-4d66-a7f1-1f3176d04621
# ╟─8a835846-2497-11eb-167d-d98f197b3895
# ╠═4a1d9e55-b0a0-4988-8173-9c33ce349430
# ╟─56fbbfb9-97d9-4f79-af2e-6d5b04c11be5
# ╠═77657312-7aab-4e91-b0d9-e362a6305fdc
# ╟─124801c2-eadb-444b-b882-0f847dd93754
# ╠═a37680d0-2557-11eb-3808-eb7d07f9c3f0
# ╟─4b7bc817-adbf-456f-92be-c6b3218a850c
# ╟─9ec36fe1-95ed-446b-b444-6a0331e64c1d
# ╟─dc5356b8-0649-4e1c-aa35-5c202ef41c36
# ╠═71f35623-5522-4d26-a4fe-c225fb0d676a
# ╠═d3422028-5f9d-4518-a48e-19a690570fa1
# ╠═bbb8abc0-457d-479c-aa8a-5883298950d3
# ╟─d5fa7759-10a8-4fd3-b15f-1abf69d19fc8
# ╟─a4bc5d3f-20e5-4796-8c60-ac2abd7e5eb0
# ╠═320c3c6d-4e8b-4197-b4af-06409e11f363
# ╟─bf635a00-7952-42ae-a855-3b7de1709c45
# ╠═1c6b141a-af6b-4967-b571-6758da010790
# ╠═928c473d-119b-4341-9ce2-04d1d2e615de
# ╟─7ab8ad79-5a46-444a-ae19-70804f9a7ae2
# ╠═8ac7e9bc-005e-442a-899a-7714b595152e
# ╠═75d6d25d-a7fc-42b1-a4bc-161018dc83ab
# ╟─6f70cbed-3386-4478-850d-0b791153c48e
# ╠═1cb15395-d9b1-4198-84f6-3e26b3dfe294
# ╠═50bc5514-0b6f-460c-babc-743fa7bba925
# ╟─707178e8-cd43-4b3b-aaf3-aafd04a2f525
# ╟─654d3af3-f3e7-4f86-a022-653acc02ddb5
# ╠═e39a4aad-0d60-41fd-9304-6cf7c6043e47
# ╠═00967516-a826-4cba-8222-4cf7d4404738
# ╠═2939982e-4a78-4087-bbd9-30241c486efb
# ╟─1a8bc38d-f4f4-452f-81f3-ca3706c7f537
# ╠═ebadc73b-ac39-4012-9332-08d2d758d2f8
# ╠═dce3578c-3170-43fe-b448-b6ffe0ec4901
# ╠═48f0a95d-f0e8-43aa-8f7c-651d9656e2af
# ╟─7b5fdf78-c154-40fe-940b-477b49bb38f8
# ╟─22ce06ce-2040-419d-996f-5fa13aab8516
