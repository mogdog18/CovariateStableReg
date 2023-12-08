using CSV, DataFrames, Statistics, Random, Plots, JuMP, Gurobi, StatsBase

using MultivariateStats

using ScikitLearn
@sk_import linear_model: LogisticRegression
@sk_import metrics:accuracy_score;
@sk_import metrics:precision_score;
@sk_import metrics:recall_score;


# ------- helper functions to load data -----------
function get_abalone_data()
    df = CSV.read("../data/abalone/abalone_original.csv", DataFrame)
    selected_columns = setdiff(names(df), ["rings", "sex"])

    X = df[:,selected_columns]
    y = CSV.read("../data/abalone/abalone_original.csv", DataFrame)[!,"rings"];
    return X, y

end

function get_comp_hard_data()
    df = CSV.read("../data/comp_hard/machine.data", header = false, DataFrame)
    df_names = ["Vendor_Name", "Model_Name", "MYCT", "MMIN", "MMAX", "CACH", "CHMIN", "CHMAX", "PRP", "ERP"]
    rename!(df, Symbol.(df_names))

    feature_list = ["MYCT", "MMIN", "MMAX", "CACH", "CHMIN", "CHMAX", "PRP"]
    X = df[:, feature_list]
    y = df[!, "ERP"];

    return X, y 
end

function get_concrete_data()
    df = CSV.read("../data/concrete/concrete_data.csv", DataFrame)
    selected_columns = setdiff(names(df), ["concrete_compressive_strength"])

    X = df[:,selected_columns]
    y = df[!,"concrete_compressive_strength"];
    return X, y
end

function get_forestfire_data()
    df = CSV.read("../data/forest_fire/forestfires.csv", DataFrame)
    X = select(df, Not(["month", "day", "area"]))

    # one hot encode months
    month_encoded = indicatormat(Matrix(select(df, "month")))
    month_encoded = transpose(month_encoded)
    month_df = DataFrame(ones(size(month_encoded)), :auto)

    for i in 1:size(month_encoded, 2)
        column_name = "x$i"
        month_df[!, column_name] = month_encoded[:, i]
    end
    new_column_names = Symbol.("month_" .* string.(1:12))
    rename!(month_df, new_column_names)

    # one hot encode days
    day_encoded = indicatormat(Matrix(select(df, "day")))
    day_encoded = transpose(day_encoded)
    day_df = DataFrame(ones(size(day_encoded)), :auto)
    for i in 1:size(day_encoded, 2)
        column_name = "x$i"
        day_df[!, column_name] = day_encoded[:, i]
    end
    new_column_names = Symbol.("day_" .* string.(1:7))
    rename!(day_df, new_column_names)

    # combine one hot encoding with original dataset
    X = hcat(X, day_df, month_df)

    # get y values
    y = CSV.read("../data/forest_fire/forestfires.csv", DataFrame)[!,"area"];
    return X, y

end
# --------- Metrics -------------
function mse(y_true, y_pred)
    return sum((y_true .- y_pred).^2) / length(y_true)
end



# ------- plotting fucntions ---------
function plot_covariate_shit(X_train, X_test_shift)
    plot_list = []
    num_cols = 2

    for (i, col) in enumerate(names(X_train))
        # Create density plots
        p = histogram(X_train[:, col], label="X_train", alpha=0.7, nbins=20, title=string("Column: ", col), color="#a3a3a3")
        histogram!(X_test_shift[:, col], label="X_test", alpha=0.7, nbins=20, color="#666666")
        
        push!(plot_list, p)
    end

    plot(plot_list..., layout=(length(plot_list) ÷ num_cols + 1, num_cols), legend=true, size=(800, 800))

end

# # ------ helper functions for preprocess data -------

function add_intercept(X)
    return hcat(ones(Int, size(X, 1)), X)
end

function normalize_data(X_train, X_test)
    # Calculate mean and standard deviation from training data
    mean_vals = Statistics.mean(X_train, dims=1)
    std_vals = Statistics.std(X_train, dims=1)

    # Normalize training data
    X_train_norm = (X_train .-mean_vals) ./ std_vals;
    X_test_norm = (X_test .-mean_vals) ./ std_vals;
    
    return X_train_norm, X_test_norm
end


# -------- function to generate data  ----------

function normal_distribution(mu, std, num_samples, seed = 1)
    Random.seed!(seed)
    return mu .+ std * randn(num_samples)
end 

function generate_synthetic_data(num_samples, type = "1", seed = 1)
    
    if type == "1"
        f_1(x) = sinc(x)

        X_train, X_test = normal_distribution(1, (1/2), num_samples, seed),  normal_distribution(2, (1/4), num_samples, seed)
        error_train, error_test = normal_distribution(0, (1/4), num_samples, seed + 1), normal_distribution(0, (1/4), num_samples, seed + 2)

        y_train = f_1.(X_train) + error_train
        y_test = f_1.(X_test) + error_test

    elseif type == "2"
        f_2(x) = -x + x^3 

        X_train, X_test = normal_distribution(0.5, 0.5, num_samples, seed), normal_distribution(0, 0.3, num_samples, seed)
        error_train, error_test = normal_distribution(0, 0.3, num_samples, seed + 1), normal_distribution(0, 0.3, num_samples, seed + 2)

        y_train = f_2.(X_train) + error_train
        y_test = f_2.(X_test) + error_test
    end
    
    return (DataFrame(Feature1 = X_train), y_train), (DataFrame(Feature1 = X_test), y_test)
end

function plot_synthetic(X, y, X_shifted, y_shifted, type = "1")
    if type == "1"
        f_1(x) = sinc(x)

        plot_obj = scatter(Matrix(X_full), y_full, label="Training Samples", xlabel="x", ylabel="f(x) + ε", color="#a3a3a3")
        scatter!(Matrix(X_shifted), y_shifted, label="Test Samples", color="#666666")

        x_values = range(-0.5, stop=2, length=500)
        plot!(x_values, f_1.(x_values), label="sinc(x)", linewidth=6, color="#cbd0f1")

        savefig(plot_obj, "../data/imgs/covariate_shift_synthetic_data_1.png")

        display(plot_obj)

    else 
        f_2(x) = -x + x^3 

        plot_obj = scatter(Matrix(X_full), y_full, label="Training Samples", xlabel="x", ylabel="f(x) + ε", color="#a3a3a3")
        scatter!(Matrix(X_shifted), y_shifted, label="Test Samples", color="#666666")

        x_values = range(-1, stop=2, length=500)
        plot!(x_values, f_2.(x_values), label="-x + x^3", linewidth=6, color="#cbd0f1")

        savefig(plot_obj, "../data/imgs/covariate_shift_synthetic_data_2.png")

        display(plot_obj)
    end 

end

function plot_synthetic_slopes(mean_opt_weights_betas, mean_random_weights_betas, mean_random_betas, mean_opt_betas, func_type)
    if func_type == "1"
        f(x) = sinc(x)
    else
        f(x) =  -x + x^3
    end

    f_ow(x) =  mean_opt_weights_betas[1] +  mean_opt_weights_betas[2]* x
    f_rw(x) =  mean_random_weights_betas[1] +  mean_random_weights_betas[2]* x
    f_r(x) =  mean_random_betas[1] +  mean_random_betas[2]* x
    f_o(x) =  mean_opt_betas[1] +  mean_opt_betas[2]* x

    plot_obj = scatter(Matrix(X_full), y_full, label="Training Samples", xlabel="x", ylabel="f(x) + ε", color="#a3a3a3")
    scatter!(Matrix(X_shifted), y_shifted, label="Test Samples", color="#666666")

    x_values = range(-0.5, stop=2, length=500)
    plot!(x_values, f.(x_values), label="sinc(x)", linewidth=3, size=(800, 600), color="#cbd0f1")
    plot!(x_values, f_ow.(x_values), label="opt with weights", linewidth=3, size=(800, 600), color="#f02937")
    plot!(x_values, f_o.(x_values), label="opt no weights", linewidth=3, size=(800, 600), color="#6b1218")
    plot!(x_values, f_rw.(x_values), label="rand with weights", linewidth=3, size=(800, 600), color="#b6d7a8")
    plot!(x_values, f_r.(x_values), label="rand no weights", linewidth=3, size=(800, 600), color="#3e6e29")

    savefig(plot_obj, "../data/imgs/covariate_shift_synthetic_data_$(func_type).png")

    display(plot_obj)
end

# function normalize_data(X_train, X_test)
#     # Calculate mean and standard deviation from training data
#     mean_vals = mean(X_train, dims=1)
#     std_vals = std(X_train, dims=1)

#     # Normalize training data
#     X_train_norm = (X_train .-mean_vals) ./ std_vals;
#     X_test_norm = (X_test .-mean_vals) ./ std_vals;

#     return X_train_norm, X_test_norm
# end


# # ------- helper functions for covariate shift data -----------

function get_weights(X_train, X_test, print_results = false)
    X_combined = vcat(X_train, X_test)
    y_combined = vcat(zeros(size(X_train)[1]),ones(size(X_test)[1]))
    lr = ScikitLearn.fit!(LogisticRegression(max_iter = 2000, random_state = 1), Matrix(X_combined), y_combined)
    if print_results
        y_pred_combined = lr.predict(Matrix(X_combined))
        check_accuracy(y_pred_combined, y_combined)
    end
    y_pred_combined_prob = lr.predict_proba(Matrix(X_train))
    weights_shift = y_pred_combined_prob[:, 2] ./ y_pred_combined_prob[:, 1]
    return weights_shift
end

function LassoRegression(X, y, lambda, weight = "nothing")
    # add column of ones
    n, p = size(X)

    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 0)

    @variable(model, u[1:n])
    @variable(model, beta[1:p])
    @variable(model, w[1:p])

    @objective(model, Min, sum(u) + lambda * sum(w))

    for i in 1:p
        @constraint(model, w[i] >= beta[i])
        @constraint(model, w[i] >= -beta[i])
    end

    if weight == "nothing"
        for i in 1:n
            @constraint(model, u[i] >= (y[i] - sum(X[i, :] .* beta)))
            @constraint(model, u[i] >=  - (y[i] - sum(X[i, :] .* beta)))
        end
    else
        for i in 1:n
            @constraint(model, u[i] >= weight[i] * (y[i] - sum(X[i, :] .* beta)))
            @constraint(model, u[i] >= - weight[i] * (y[i] - sum(X[i, :] .* beta)))
        end
    end

    optimize!(model)

    return value.(beta)
end



# ------- PCA ---------
function perform_pca(dataset)
    # train a PCA model,
    pca = StatsBase.fit(PCA, dataset'; maxoutdim=size(dataset, 2))
    # apply PCA model to data set
    pca_result = transpose(MultivariateStats.transform(pca, dataset'))
    return pca_result
end



function calculate_avg_betas(rand_betas, rand_weights_betas, opt_betas, opt_weights_betas)
    mean_random_betas = mean(rand_betas, dims = 1)[1]
    mean_random_weights_betas = mean(rand_weights_betas, dims = 1)[1]
    mean_opt_betas = mean(opt_betas, dims = 1)[1]
    mean_opt_weights_betas = mean(opt_weights_betas, dims = 1)[1]
    return mean_random_betas, mean_random_weights_betas, mean_opt_betas, mean_opt_weights_betas
end


# function generate_covariate_shift(X_test)
#     #set a seed for distribution
#     Random.seed!(123)

#     col_means = mean(Matrix(X_test), dims=1)
#     std_dev = 0.1

#     random_values = zeros(size(X_test))
#     for col_index in 1:size(X_test, 2)
#         col_mean = col_means[col_index]
#         col_shift = rand(Normal(col_mean * 0.5 , std_dev), (size(X_test, 1), 1))
#         random_values[:, col_index] = col_shift
#     end

#     X_test_shift = Matrix(X_test) .+ random_values;
#     df_X_test_shift = DataFrame(X_test_shift,Symbol.(names(X_test)))
#     return df_X_test_shift
# end

# function get_weights(X_train, X_test, print_results = false)
#     X_combined = vcat(X_train, X_test)
#     y_combined = vcat(zeros(size(X_train)[1]),ones(size(X_test)[1]))
#     lr = fit!(LogisticRegression(max_iter = 2000, random_state = 1), Matrix(X_combined), y_combined)
#     if print_results
#         y_pred_combined = lr.predict(Matrix(X_combined))
#         check_accuracy(y_pred_combined, y_combined)
#     end
#     y_pred_combined_prob = lr.predict_proba(Matrix(X_train))
#     weights_shift = y_pred_combined_prob[:, 2] ./ y_pred_combined_prob[:, 1]
#     return weights_shift
# end


# # ------ helper functions for stable regression -------- 
# function optimized_split(X, y, lambda, train_fraction, weights = nothing)
#     # add column of ones
#     X = hcat(ones(Int, size(X, 1)), X)

#     #parameters
#     n, p = size(X)
#     k = n * train_fraction

#     model = Model(Gurobi.Optimizer)
#     set_optimizer_attribute(model, "OutputFlag", 0)

#     @variable(model, theta)
#     @variable(model, u[1:n] >= 0)
#     @variable(model, beta[1:p])
#     @variable(model, w[1:p])

#     @objective(model, Min, k * theta + sum(u) + lambda * sum(w))

#     for i in 1:p
#         @constraint(model, w[i] >= beta[i])
#         @constraint(model, w[i] >= -beta[i])
#     end 

#     for i in 1:n
#         if weights == nothing
#             @constraint(model, theta + u[i] >= y[i] - sum(X[i, :].*beta))
#             @constraint(model, theta + u[i] >= -(y[i] - sum(X[i, :].*beta)))
#         else 
#             @constraint(model, theta + u[i] >=  weights[i] * (y[i] - sum(X[i, :].*beta)))
#             @constraint(model, theta + u[i] >= - weights[i] * (y[i] - sum(X[i, :].*beta)))
#         end 
#     end
    
#     optimize!(model)

#     return value(theta), value.(u), value.(beta), value.(w)
# end

# function LasssoRegression(X, y, lambda, weight = nothing)
#     # add column of ones
#     X = hcat(ones(Int, size(X, 1)), X)
#     n, p = size(X)

#     model = Model(Gurobi.Optimizer)
#     set_optimizer_attribute(model, "OutputFlag", 0)

#     @variable(model, u[1:n])
#     @variable(model, beta[1:p])
#     @variable(model, w[1:p])

#     @objective(model, Min, sum(u) + lambda * sum(w))

#     for i in 1:p
#         @constraint(model, w[i] >= beta[i])
#         @constraint(model, w[i] >= -beta[i])
#     end

#     if weight !== nothing
#         for i in 1:n
#             @constraint(model, u[i] >= weight[i] * (y[i] - sum(X[i, :] .* beta)))
#             @constraint(model, u[i] >= - weight[i] * (y[i] - sum(X[i, :] .* beta)))
#         end
#     else
#         for i in 1:n
#             @constraint(model, u[i] >= (y[i] - sum(X[i, :] .* beta)))
#             @constraint(model, u[i] >=  - (y[i] - sum(X[i, :] .* beta)))
#         end
#     end

#     optimize!(model)

#     return value.(beta)
# end


# # ------- Metrics -----------------
function calculate_percentage_improvement(scores_after, scores_before)
    if length(scores_after) != length(scores_before)
        error("Old scores and new scores must have the same length.")
    end
    
    return mean((scores_before .- scores_after) ./scores_before) * 100
end

function check_accuracy(y_pred, y_true)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    
    println("accuracy: ", round(accuracy, digits=10))
    println("precision: ", round(precision, digits=10))
    println("recall: ", round(recall, digits=10))
    return 
end