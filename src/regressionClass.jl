using CSV, DataFrames, Statistics, Random, Plots, JuMP, Gurobi, Distributions, Random, Metrics
using StatsBase: sample

include("utils.jl");

# ------- data structure -----------

struct RegressionClass
    name::String
    X_full::DataFrame
    X_shifted::DataFrame
    y_full::Vector{Float64}
    y_shifted::Vector{Float64}
    covar_dist_type::String
    covar_parameters::Vector{Any}
    lambas_range::Vector{Float64}
    train_test_prop::Float64
    train_val_prop::Float64
    num_runs::Float64
end

# -------- preprocessing function ----------------

function perform_covariate_shift(rc::RegressionClass, X_full_norm)
    pca_result = perform_pca(X_full_norm)
    m = median(pca_result)
    X_train, X_test = [], []
    y_train, y_test = [], []

    for i in 1:size(X_full_norm, 1)
        if pca_result[i, 1] >= m
            if rand() <= rc.covar_parameters[1]
                push!(X_train, X_full_norm[i, :])
                push!(y_train, rc.y_full[i])
            else
                push!(X_test, X_full_norm[i, :])
                push!(y_test, rc.y_full[i])
            end
        end
        
        if pca_result[i, 1] < m
            if rand() <= rc.covar_parameters[2]
                push!(X_train, X_full_norm[i, :])
                push!(y_train, rc.y_full[i])
            else
                push!(X_test, X_full_norm[i, :])
                push!(y_test, rc.y_full[i])
            end
        end
    end
    
    # reshape list of lists to be a matrix
    X_train_reshaped = zeros(length(X_train), size(X_full_norm, 2))
    X_test_reshaped = zeros(length(X_test), size(X_full_norm, 2))
    
    for i in 1:size(X_train, 1)
        X_train_reshaped[i, :] = X_train[i]
    end
    for i in 1:size(X_test, 1)
        X_test_reshaped[i, :] = X_test[i]
    end
    
    idx_train = randperm(size(X_train, 1))[1:Int(rc.covar_parameters[3])]
    idx_test = randperm(size(X_test, 1))[1:Int(rc.covar_parameters[4])]
    X_train_reshaped = X_train_reshaped[idx_train, :]
    X_test_reshaped = X_test_reshaped[idx_test, :]
    y_train = y_train[idx_train]
    y_test = y_test[idx_test]
    
    return (X_train_reshaped, y_train), (X_test_reshaped, y_test)
end


function normalise_X_full(rc::RegressionClass)
    mean_vals = mean(Matrix(rc.X_full), dims=1)
    std_vals = std(Matrix(rc.X_full), dims=1)
    X_full_norm = (Matrix(rc.X_full) .-mean_vals) ./ std_vals;
    return X_full_norm 
    
end


function train_test_split(rc::RegressionClass, random_seed)
    
    Random.seed!(random_seed)

    num_indices = round(Int, rc.train_test_prop * length(rc.y_full))
    train_indices = sample(1:length(rc.y_full), num_indices, replace=false)
    test_indices = setdiff(1:length(rc.y_full), train_indices)
    
    X_train, y_train = Matrix(rc.X_full)[train_indices, :], rc.y_full[train_indices]
    X_test, y_test = Matrix(rc.X_shifted)[test_indices, :], rc.y_shifted[test_indices]
    
    return (X_train, y_train), (X_test, y_test)
end


function train_val_split(rc::RegressionClass, X_full_train, y_full_train, weights = "nothing", random_seed = 1)
    
    Random.seed!(random_seed)

    num_indices = round(Int, rc.train_val_prop * length(y_full_train))
    train_indices = sample(1:length(y_full_train), num_indices, replace=false)
    val_indices = setdiff(1:length(y_full_train), train_indices)
    
    X_train, y_train = Matrix(X_full_train)[train_indices, :],y_full_train[train_indices]
    X_val, y_val = Matrix(X_full_train)[val_indices, :], y_full_train[val_indices]

    if weights !== "nothing"
        weights = weights[train_indices]
    end
    
    return (X_train, y_train, weights), (X_val, y_val)
end


# --------Stable regression + covariate shift functions  ----------------

function train_val_opt_split(rc::RegressionClass, X_train_full, y_train_full, beta_opt, weights = "nothing")

    residuals = y_train_full - X_train_full * beta_opt

    if weights == "nothing"
        sorted_indices = sortperm(abs.(residuals), rev=true)
    else 
        residuals_weights = [residuals[i] * weights[i] for i in 1:length(weights)]
        sorted_indices = sortperm(abs.(residuals_weights), rev=true)
    end

    num_train_points = round(Int, rc.train_val_prop * length(sorted_indices))

    train_indices = sorted_indices[1:num_train_points]

    val_indices = setdiff(1:length(y_train_full), train_indices)

    X_train = X_train_full[train_indices, :] 
    y_train = y_train_full[train_indices]

    X_val = X_train_full[val_indices, :]
    y_val = y_train_full[val_indices]

    return X_train, y_train, X_val, y_val
end


function get_optimized_split(rc::RegressionClass, X_train_full, y_train_full, lambda, weights="nothing")
    # to do: should n be an attribute
    n, p = size(X_train_full)
    k = round(Int, n * rc.train_val_prop)
    
    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "OutputFlag", 0)

    @variable(model, theta)
    @variable(model, u[1:n] >= 0)
    @variable(model, beta[1:p])
    @variable(model, w[1:p])
    @objective(model, Min, k * theta + sum(u) + lambda * sum(w))
    
    for i in 1:p
        @constraint(model, w[i] >= beta[i])
        @constraint(model, w[i] >= -beta[i])
    end 
    
    for i in 1:n
        if weights == "nothing"
            @constraint(model, theta + u[i] >= y_train_full[i] - sum(X_train_full[i, :].*beta))
            @constraint(model, theta + u[i] >= -(y_train_full[i] - sum(X_train_full[i, :].*beta)))
        else 
            @constraint(model, theta + u[i] >=  weights[i] * (y_train_full[i] - sum(X_train_full[i, :].*beta)))
            @constraint(model, theta + u[i] >= - weights[i] * (y_train_full[i] - sum(X_train_full[i, :].*beta)))
        end 
    end
    
    
    optimize!(model)
    return value(theta), value.(u), value.(beta), value.(w) 
end


function get_optimized_split_test_score(rc::RegressionClass, X_full_train, y_full_train, X_test, y_test, weights="nothing", print_result=false)
    best_lambda = Inf
    best_val_mse = Inf
    best_model = Inf
    
    for lambda in rc.lambas_range
        # Get optimized split
        _, _, betas, _ = get_optimized_split(rc, X_full_train, y_full_train, lambda, weights)
        X_train, y_train, X_val, y_val = train_val_opt_split(rc, X_full_train, y_full_train, betas, weights)
        
        # Predict on validation set
        y_pred_val =  X_val * betas
        val_mse_i = mse(y_val, y_pred_val)
        
        if best_val_mse > val_mse_i
            best_lambda = lambda
            best_val_mse = val_mse_i
            best_model = betas
        end
    end

    #get mse on test set for best performing model 
    y_pred_test = X_test * best_model
    mse_test_mse = mse(y_test, y_pred_test)

    if print_result
        println("Best lambda: ", best_lambda)
        println("Validation score: ", best_val_mse)
        println("Test score: ", mse_test_mse)
        println("Number of betas: ", length(best_model))
    end

    return best_model, mse_test_mse
end

function get_random_test_score(rc::RegressionClass, X_full_train, y_full_train, X_test, y_test, seed_value, weights="nothing", print_result=false)
    best_lambda = Inf
    best_val_mse = Inf
    best_model = Inf

    (X_train, y_train, weights_for_train), (X_val, y_val) = train_val_split(rc, X_full_train, y_full_train, weights, seed_value)

    for lambda in rc.lambas_range
        
        beta_star = LassoRegression(X_train, y_train, lambda, weights_for_train)

        y_pred_val = X_val * beta_star
        val_mse_i = mse(y_val, y_pred_val)

        if best_val_mse > val_mse_i
            best_lambda = lambda
            best_val_mse = val_mse_i
            best_model = beta_star
        end
    end

    #get test mse score on best performing model
    y_pred_test = X_test * best_model
    mse_test_mse = mse(y_test, y_pred_test)

    if print_result
        println("Best lambda: ", best_lambda)
        println("Validation score: ", best_val_mse)
        println("Test score: ", mse_test_mse)
        println("Number of betas: ", length(best_model))
    end
   
    return best_model, mse_test_mse
end

#todo: add what is returned to instance 
function repeat_four_methods(rc::RegressionClass)

    random_mse_test_scores, random_weights_mse_test_scores = [], []
    optim_mse_test_scores, optim_weights_mse_test_scores = [], []

    rand_betas, rand_weights_betas = [], []
    opt_betas, opt_weights_betas = [], []

    for random_seed in 1:rc.num_runs
        random_seed = Int(random_seed) #need to be int 

        if rc.covar_dist_type == "PCA"
            #pca shift on real dataset 
            X_full_norm = normalise_X_full(rc)
            (X_train, y_train), (X_test, y_test) = perform_covariate_shift(rc, X_full_norm)
        else
            # synthetic data (shift has already been performed when creating the dataset)
            (X_train, y_train), (X_test, y_test) = train_test_split(rc, random_seed)
        end

        X_train_norm, X_test_norm = normalize_data(X_train, X_test)
        X_train_norm, X_test_norm = add_intercept(X_train_norm), add_intercept(X_test_norm)

        weights = get_weights(X_train_norm, X_test_norm, false)

        if random_seed % 10 == 0
            println(random_seed)
        end

        # println("Starting Randomization")
        rand_beta, random_mse_test_score = get_random_test_score(rc, X_train_norm, y_train, X_test_norm, y_test, random_seed, "nothing", false)
        push!(random_mse_test_scores, random_mse_test_score)
        push!(rand_betas, rand_beta)
        
        
        # println("Starting Randomization with covariate weights")
        rand_weights_beta, random_weights_mse_test_score = get_random_test_score(rc, X_train_norm, y_train, X_test_norm, y_test, random_seed, weights, false)
        push!(random_weights_mse_test_scores, random_weights_mse_test_score)
        push!(rand_weights_betas, rand_weights_beta)
        
        # println("Starting Optimization")
        opt_beta, optim_mse_test_score = get_optimized_split_test_score(rc, X_train_norm, y_train, X_test_norm, y_test, "nothing", false)
        push!(optim_mse_test_scores, optim_mse_test_score)
        push!(opt_betas, opt_beta)

        
        # println("Starting Optimization with covariate weights")
        opt_weights_beta, optim_weights_mse_test_score = get_optimized_split_test_score(rc, X_train_norm, y_train, X_test_norm, y_test, weights, false)
        push!(optim_weights_mse_test_scores, optim_weights_mse_test_score)
        push!(opt_weights_betas, opt_weights_beta)
    end 
    
    return (rand_betas, rand_weights_betas, opt_betas, opt_weights_betas), (random_mse_test_scores, random_weights_mse_test_scores, optim_mse_test_scores, optim_weights_mse_test_scores)
end


