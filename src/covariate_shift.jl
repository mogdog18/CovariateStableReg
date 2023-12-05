using CSV, DataFrames, Statistics, Random, Plots, JuMP, Gurobi, Distributions, Random, Metrics
using StatsBase: sample
 
#import sklearn functions 
using ScikitLearn
@sk_import linear_model: LogisticRegression
@sk_import linear_model: LinearRegression
@sk_import metrics:accuracy_score;
@sk_import metrics:precision_score;
@sk_import metrics:recall_score;
@sk_import metrics:mean_squared_error;

#import own functions
include("utils.jl")


struct CovariateShift
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

function train_test_split(rc::CovariateShift, random_seed)
    
    Random.seed!(random_seed)

    num_indices = round(Int, rc.train_test_prop * length(rc.y_full))
    train_indices = sample(1:length(rc.y_full), num_indices, replace=false)
    test_indices = setdiff(1:length(rc.y_full), train_indices)
    
    X_train, y_train = Matrix(rc.X_full)[train_indices, :], rc.y_full[train_indices]
    X_test, y_test = Matrix(rc.X_shifted)[test_indices, :], rc.y_shifted[test_indices]
    
    return (X_train, y_train), (X_test, y_test)
end

function get_random_test_score(rc::CovariateShift, X_full_train, y_full_train, X_test, y_test, seed_value, weights="nothing", print_result=false)
    best_lambda = Inf
    best_val_mse = Inf
    best_model = Inf

    (X_train, y_train), (X_val, y_val) = 
        IAI.split_data(:regression, X_full_train, y_full_train, train_proportion=rc.train_val_prop, seed=seed_value)

    for lambda in rc.lambas_range
        beta_star = LassoRegression(X_train, y_train, lambda, weights)

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
        println("Number of betas: ", length(best_model.coef_) + length(best_model.intercept_))
    end
   
    return mse_test_mse
end

#todo: deal with prints (false)
#todo: add what is returned to instance 
function repeat_four_methods(rc::CovariateShift)

    random_mse_test_scores, random_weights_mse_test_scores = [], []

    rand_betas, rand_weights_betas = [], []

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
    end 
    
    return (rand_betas, rand_weights_betas, opt_betas, opt_weights_betas), (random_mse_test_scores, random_weights_mse_test_scores, optim_mse_test_scores, optim_weights_mse_test_scores)
end


#hyperparameters
num_runs = 100
train_test_prop = 0.9
train_val_prop = 0.7
lambdas = [0.00001, 0.0001, 0.001, 0.01, 0.1];

(X_full, y_full), (X_shifted, y_shifted) = generate_synthetic_data(500, "2", 123)

synthetic_instance_2 = CovariateShift(
    "synthetic_2",
    X_full,
    X_shifted,
    y_full,
    y_shifted,
    "Synthetic",
    lambdas,
    [],
    train_test_prop,
    train_val_prop,
    num_runs);

(rand_betas, rand_weights_betas), (random_mse_test_scores, random_weights_mse_test_scores) = repeat_four_methods(synthetic_instance_2);

print(mean((random_mse_test_scores .- random_weights_mse_test_scores) ./random_mse_test_scores * 100))
