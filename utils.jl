using CSV, DataFrames, Statistics, Random, Plots, JuMP, Gurobi

function get_abalone_data()
    df = CSV.read("data/abalone/abalone_original.csv", DataFrame)
    selected_columns = setdiff(names(df), ["rings", "sex"])

    X = df[:,selected_columns]
    y = CSV.read("data/abalone/abalone_original.csv", DataFrame)[!,"rings"];
    return X, y

end

function get_comp_hard_data()
    df = CSV.read("data/comp_hard/machine.data", header = false, DataFrame)
    df_names = ["Vendor_Name", "Model_Name", "MYCT", "MMIN", "MMAX", "CACH", "CHMIN", "CHMAX", "PRP", "ERP"]
    rename!(df, Symbol.(df_names))

    feature_list = ["MYCT", "MMIN", "MMAX", "CACH", "CHMIN", "CHMAX", "PRP"]
    X = df[:, feature_list]
    y = df[!, "ERP"];

    return X, y 
end

function get_concrete_data()
    df = CSV.read("data/concrete/concrete_data.csv", DataFrame)
    selected_columns = setdiff(names(df), ["concrete_compressive_strength"])

    X = df[:,selected_columns]
    y = df[!,"concrete_compressive_strength"];
    return X, y
end