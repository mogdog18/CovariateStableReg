using CSV, DataFrames, Statistics, Random, Plots, JuMP, Gurobi

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
        p = histogram(X_train[:, col], label="X_train", alpha=0.7, nbins=20,title=string("Column: ", col))
        histogram!(X_test_shift[:, col], label="X_test", alpha=0.7, nbins=20)
        
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
    mean_vals = mean(X_train, dims=1)
    std_vals = std(X_train, dims=1)

    # Normalize training data
    X_train_norm = (X_train .-mean_vals) ./ std_vals;
    X_test_norm = (X_test .-mean_vals) ./ std_vals;
    
    return X_train_norm, X_test_norm
end


# -------- function to generate data  ----------
function generate_synthetic_data(num_samples)

    Random.seed!(123)  # Setting seed for reproducibility
    f(x) = sinc(x)
    normal_dist(mu, std, num_samples) = mu .+ std * randn(num_samples)

    X_train, X_test = normal_dist(1, 1/2, num_samples),  normal_dist(2, 1/4, num_samples)
    error_train, error_test = normal_dist(0, 1/4, num_samples), normal_dist(0, 1/4, num_samples) # ε with density φ(ε;0,(1/4)^2)

    y_train = f.(X_train) + error_train
    y_test = f.(X_test) + error_test

    return (DataFrame(Feature1 = X_train), y_train), (DataFrame(Feature1 = X_test), y_test)
end



function generate_synthetic_data_2(num_samples, shiftfactor = 0.8)
    Random.seed!(123)  
    
    f(x) = -x[:, 1].^2 .+ x[:, 2] .- 1
    
    normal_dist(mu, std, num_samples) = mu .+ std * randn(num_samples)

    X_train, X_test = randn(num_samples, 2), randn(num_samples, 2) .+ shiftfactor
    error_train, error_test = normal_dist(0, 1/4, num_samples), normal_dist(0, 1/4, num_samples) # ε with density φ(ε;0,(1/4)^2)

    y_train = f(X_train) + error_train
    y_test = f(X_test) + error_test

    return (DataFrame(Feature1 = X_train[:, 1], Feature2 = X_train[:, 2]), y_train),
           (DataFrame(Feature1 = X_test[:, 1], Feature2 = X_test[:, 2]), y_test)
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
    lr = fit!(LogisticRegression(max_iter = 2000, random_state = 1), Matrix(X_combined), y_combined)
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

    if weight !== "nothing"
        for i in 1:n
            @constraint(model, u[i] >= weight[i] * (y[i] - sum(X[i, :] .* beta)))
            @constraint(model, u[i] >= - weight[i] * (y[i] - sum(X[i, :] .* beta)))
        end
    else
        for i in 1:n
            @constraint(model, u[i] >= (y[i] - sum(X[i, :] .* beta)))
            @constraint(model, u[i] >=  - (y[i] - sum(X[i, :] .* beta)))
        end
    end

    optimize!(model)

    return value.(beta)
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
function calculate_percentage_improvement(opt_scores, rand_scores)
    if length(opt_scores) != length(rand_scores)
        error("Old scores and new scores must have the same length.")
    end
    
    percentage_improvements = []

    for i in 1:length(opt_scores)
        opt_score = opt_scores[i]
        rand_score = rand_scores[i]

        percentage_improvement = ((rand_score - opt_score) / rand_score) * 100
        push!(percentage_improvements, percentage_improvement)
    end

    return mean(percentage_improvements)
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