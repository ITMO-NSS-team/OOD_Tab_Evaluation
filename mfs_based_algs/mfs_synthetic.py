import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.datasets import make_classification
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from pymfe.mfe import MFE
from deap import base, creator, tools, algorithms
from ForestDiffusion import ForestDiffusionModel
import random
import warnings
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA
from functools import partial
import json
from multiprocessing import Pool
import glob
from sklearn.preprocessing import LabelEncoder
from deap.tools._hypervolume import pyhv
from deap.tools.emo import uniform_reference_points

plt.style.use("seaborn-v0_8-whitegrid")
warnings.filterwarnings("ignore")

# Clear any existing DEAP creator classes to avoid conflicts
if hasattr(creator, "FitnessMulti"):
    del creator.FitnessMulti
if hasattr(creator, "Individual"):
    del creator.Individual

# Create DEAP classes for multi-objective optimization
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

# Global variables for storing meta-feature values
init_val = "a"
fin_val = "a"
tar_val = "a"
fit_value = float("inf")


def compute_meta_feature(data, meta_feature):
    """
    Compute the value of a specific meta-feature for given data.

    Parameters:
    -----------
    data : numpy.ndarray or DataFrame
        Data to compute meta-feature for
    meta_feature : str
        Name of the meta-feature to compute

    Returns:
    --------
    float
        Computed meta-feature value
    """
    try:
        mfe = MFE(features=[meta_feature], summary=None)

        # Make sure data is in numpy array format
        if isinstance(data, pd.DataFrame):
            data = data.values

        # Ensure data is not empty and has at least 2 dimensions
        if data.size == 0 or data.ndim < 2:
            return np.nan

        # Check if there's enough data for X and y
        if data.shape[1] < 2:
            return np.nan

        # Ensure data is float type and replace any NaN values
        data = np.array(data)
        data = np.nan_to_num(data)

        X = data[:, :-1]
        y = data[:, -1]

        # Check if there's any variation in the target
        if len(np.unique(y)) < 2:
            return 0.0

        mfe.fit(X, y)
        ft = mfe.extract()[1][0]
        return ft

    except Exception as e:
        print(f"Error computing {meta_feature}: {e}")
        return np.nan


def mutate_noise(
    individual,
    mutation_prob=0.3,
    noise_scale=0.1,
    categorical_idx=None,
    continuous_idx=None,
    cat_probs=None,
    n_features=None,
):
    """
    Row-wise mutation: Add Gaussian noise to individual rows.
    """
    # Reshape individual to 2D
    individual_data = np.array(individual).reshape(-1, n_features)
    mutated = individual_data.copy()

    # Mutate each row with probability mutation_prob
    for i in range(len(individual_data)):
        if random.random() > mutation_prob:
            continue

        # Add noise to continuous features
        if continuous_idx:
            # Scale noise relative to the data values
            noise_scale_adjusted = noise_scale * np.abs(mutated[i][continuous_idx])
            # Avoid zero noise scale
            noise_scale_adjusted = np.maximum(noise_scale_adjusted, 0.01)
            noise = np.random.normal(0, noise_scale_adjusted)

            # Ensure noise has correct dimensions
            if not isinstance(noise, np.ndarray) or len(noise) != len(continuous_idx):
                noise = np.random.normal(0, 0.1, len(continuous_idx))

            mutated[i][continuous_idx] += noise

        # Handle categorical features
        if categorical_idx and cat_probs:
            for j, cat_idx in enumerate(categorical_idx):
                if j < len(cat_probs) and random.random() < mutation_prob:
                    mutated[i, cat_idx] = np.random.choice(
                        len(cat_probs[j]), p=cat_probs[j]
                    )

    # Replace any NaN values
    mutated = np.nan_to_num(mutated)
    if categorical_idx:
        for cat_idx in categorical_idx:
            # Round to nearest integer
            mutated[:, cat_idx] = np.round(mutated[:, cat_idx]).astype(int)
            # Ensure values are valid categories (assuming binary 0/1 classification)
            mutated[:, cat_idx] = np.clip(mutated[:, cat_idx], 0, 1)
    return (creator.Individual(mutated.flatten().tolist()),)


def mutate_dist(
    individual,
    mutation_prob=0.3,
    gmm=None,
    categorical_idx=None,
    continuous_idx=None,
    cat_probs=None,
    n_features=None,
):
    """
    Row-wise mutation: Generate new values from marginal distributions.
    """
    # Reshape individual to 2D
    individual_data = np.array(individual).reshape(-1, n_features)
    mutated = individual_data.copy()

    # Mutate each row with probability mutation_prob
    for i in range(len(individual_data)):
        if random.random() > mutation_prob:
            continue

        # Generate new categorical values
        new_categorical = []
        if categorical_idx and cat_probs:
            new_categorical = [np.random.choice(len(p), p=p) for p in cat_probs]

        # Generate new continuous values
        new_continuous = []
        if continuous_idx and gmm:
            try:
                new_continuous = gmm.sample(1)[0].flatten()
                if len(new_continuous) > len(continuous_idx):
                    new_continuous = new_continuous[: len(continuous_idx)]
                elif len(new_continuous) < len(continuous_idx):
                    # Pad with zeros or repeat values if needed
                    new_continuous = np.pad(
                        new_continuous,
                        (0, len(continuous_idx) - len(new_continuous)),
                        mode="constant",
                    )
            except Exception as e:
                print(f"Error in GMM sampling: {e}")
                # Fallback to using random normal distribution
                mean = np.mean(individual_data[:, continuous_idx], axis=0)
                std = np.std(individual_data[:, continuous_idx], axis=0)
                new_continuous = np.random.normal(mean, std)

        # Assemble new row
        if categorical_idx and continuous_idx:
            new_row = np.zeros(n_features)
            for j, idx in enumerate(categorical_idx):
                if j < len(new_categorical):
                    new_row[idx] = new_categorical[j]
            for j, idx in enumerate(continuous_idx):
                if j < len(new_continuous):
                    new_row[idx] = new_continuous[j]
        elif categorical_idx:
            new_row = np.zeros(n_features)
            for j, idx in enumerate(categorical_idx):
                if j < len(new_categorical):
                    new_row[idx] = new_categorical[j]
        else:
            new_row = np.zeros(n_features)
            for j, idx in enumerate(continuous_idx):
                if j < len(new_continuous):
                    new_row[idx] = new_continuous[j]

        mutated[i] = new_row

    # Replace any NaN values
    mutated = np.nan_to_num(mutated)
    if categorical_idx:
        for cat_idx in categorical_idx:
            # Round to nearest integer
            mutated[:, cat_idx] = np.round(mutated[:, cat_idx]).astype(int)
            # Ensure values are valid categories (assuming binary 0/1 classification)
            mutated[:, cat_idx] = np.clip(mutated[:, cat_idx], 0, 1)
    return (creator.Individual(mutated.flatten().tolist()),)


def mutate_all(
    individual,
    mutation_prob=0.3,
    gmm=None,
    categorical_idx=None,
    continuous_idx=None,
    cat_probs=None,
    n_features=None,
):
    """
    Row-wise mutation that combines all strategies
    """
    # Reshape individual to 2D
    individual_data = np.array(individual).reshape(-1, n_features)
    mutated = individual_data.copy()

    # Random choice of mutation type
    mutation_type = random.choices(
        ["dist", "noise", "cov"], weights=[0.5, 0.3, 0.2], k=1
    )[0]

    # Prepare parameters
    mutation_params = {
        "mutation_prob": mutation_prob,
        "categorical_idx": categorical_idx,
        "continuous_idx": continuous_idx,
        "cat_probs": cat_probs,
        "n_features": n_features,
    }

    # Apply selected mutation
    if mutation_type == "dist":
        mutation_params["gmm"] = gmm
        return mutate_dist(individual, **mutation_params)
    elif mutation_type == "noise":
        return mutate_noise(individual, noise_scale=0.1, **mutation_params)
    else:  # cov
        return mutate_cov(individual, **mutation_params)


def mutate_cov(
    individual,
    mutation_prob=0.15,
    categorical_idx=None,
    continuous_idx=None,
    cat_probs=None,
    n_features=None,
):
    """
    Row-wise mutation: Generate new values based on covariance matrix.
    """
    # Reshape individual to 2D
    individual_data = np.array(individual).reshape(-1, n_features)
    mutated = individual_data.copy()

    try:
        # Calculate covariance matrix for continuous features
        if continuous_idx:
            continuous_data = individual_data[:, continuous_idx]
            # Ensure data is numerical and non-empty
            if continuous_data.size > 0 and not np.isnan(continuous_data).any():
                # Compute covariance matrix
                current_cov = np.cov(continuous_data.T)

                # Ensure it's a 2D matrix (for single feature case)
                if current_cov.ndim == 0:
                    current_cov = np.array([[current_cov]])

                # Ensure positive definiteness
                try:
                    min_eig = np.min(np.linalg.eigvals(current_cov))
                    if min_eig < 0:
                        current_cov -= 10 * min_eig * np.eye(*current_cov.shape)
                except np.linalg.LinAlgError:
                    current_cov += 0.01 * np.eye(*current_cov.shape)

                # Generate new samples for continuous features using multivariate normal
                try:
                    mean_vector = np.mean(continuous_data, axis=0)
                    # Generate new data
                    new_continuous = np.random.multivariate_normal(
                        mean_vector, current_cov, size=len(individual_data)
                    )

                    # Apply with mutation probability
                    for i in range(len(individual_data)):
                        if random.random() <= mutation_prob:
                            # Update continuous features
                            mutated[i, continuous_idx] = new_continuous[i]
                except Exception as e:
                    print(f"Error generating samples from covariance matrix: {e}")
                    # Fallback to simple noise
                    for i in range(len(individual_data)):
                        if random.random() <= mutation_prob:
                            mutated[i, continuous_idx] += np.random.normal(
                                0, 0.1, len(continuous_idx)
                            )

        # Handle categorical features
        if categorical_idx and cat_probs:
            for i in range(len(individual_data)):
                if random.random() <= mutation_prob:
                    for j, cat_idx in enumerate(categorical_idx):
                        if j < len(cat_probs):
                            mutated[i, cat_idx] = np.random.choice(
                                len(cat_probs[j]), p=cat_probs[j]
                            )

    except Exception as e:
        print(f"Error in covariance mutation: {e}")
        # In case of error, add slight noise
        for i in range(len(individual_data)):
            if random.random() <= mutation_prob:
                mutated[i] += np.random.normal(0, 0.01, n_features)

    # Replace any NaN values
    mutated = np.nan_to_num(mutated)
    if categorical_idx:
        for cat_idx in categorical_idx:
            # Round to nearest integer
            mutated[:, cat_idx] = np.round(mutated[:, cat_idx]).astype(int)
            # Ensure values are valid categories (assuming binary 0/1 classification)
            mutated[:, cat_idx] = np.clip(mutated[:, cat_idx], 0, 1)
    return (creator.Individual(mutated.flatten().tolist()),)


def crossover(ind1, ind2, cxpb=0.6, row_mode_prob=0.5, n_features=None):
    """
    Custom crossover that can work on rows or columns.
    """
    if random.random() >= cxpb:
        return ind1, ind2

    # Reshape to 2D
    matrix1 = np.array(ind1).reshape(-1, n_features)
    matrix2 = np.array(ind2).reshape(-1, n_features)
    n_samples = matrix1.shape[0]

    # Choose crossover type (row or column)
    if random.random() < row_mode_prob:
        # Row-wise crossover
        # Select random rows to exchange
        n_rows = int(n_samples * 0.3)  # Exchange about 30% of rows
        rows = random.sample(range(n_samples), k=n_rows)

        # Exchange rows
        temp = matrix1[rows].copy()
        matrix1[rows] = matrix2[rows]
        matrix2[rows] = temp
    else:
        # Column-wise crossover
        # Select random columns to exchange
        n_cols = int(n_features * 0.3)  # Exchange about 30% of columns
        cols = random.sample(range(n_features), k=n_cols)

        # Exchange columns
        temp = matrix1[:, cols].copy()
        matrix1[:, cols] = matrix2[:, cols]
        matrix2[:, cols] = temp

    # Replace any NaN values
    matrix1 = np.nan_to_num(matrix1)
    matrix2 = np.nan_to_num(matrix2)

    # Convert back to flattened list
    ind1[:] = matrix1.flatten().tolist()
    ind2[:] = matrix2.flatten().tolist()

    return ind1, ind2


def fitness_function_multi(
    individual, meta_features, target_values, n_features, initial_data=None
):
    """
    Multi-objective fitness function returning multiple fitness values for vector meta-features.
    """
    global init_val, fin_val, tar_val, fit_value

    # Reshape individual to match data dimensions
    synthetic_data = np.array(individual).reshape(-1, n_features)
    synthetic_data = np.nan_to_num(synthetic_data)

    fitness_values = []
    for meta_feature in meta_features:
        # Compute meta-feature values as vector
        synthetic_value = compute_meta_feature(synthetic_data, meta_feature)
        target_value = target_values[meta_feature]

        # Calculate Euclidean distance between vectors
        fitness = np.linalg.norm(synthetic_value - target_value)
        fitness_values.append(fitness)

    # For tracking purposes
    total_fitness = sum(fitness_values)
    if total_fitness < fit_value:
        fin_val = {mf: compute_meta_feature(synthetic_data, mf) for mf in meta_features}
        fit_value = total_fitness

    return tuple(fitness_values)


def select_solution_from_pareto(pareto_front, strategy="balanced"):
    """
    Select a solution from the Pareto front based on different strategies
    """
    if len(pareto_front) == 0:
        return None

    if strategy == "balanced":
        # Select solution with best compromise between objectives
        fitness_values = np.array([ind.fitness.values for ind in pareto_front])
        normalized_fitness = (fitness_values - fitness_values.min(axis=0)) / (
            fitness_values.max(axis=0) - fitness_values.min(axis=0)
        )
        distances = np.linalg.norm(normalized_fitness, axis=1)
        best_idx = np.argmin(distances)
        print(f"Best index: {best_idx}")
        return pareto_front[best_idx]

    return pareto_front[0]


def generate_synthetic_data(
    mutation_type,
    source_data,
    meta_features,
    target_meta_values,
    mutation_prob=0.4,
    crossover_prob=0.5,
    row_mode_prob=0.5,
    population_size=2,
    generations=2,
    forest_diffusion=None,
):
    """
    Generate synthetic data using NSGA-II optimization to match multiple meta-feature vectors.
    """
    global init_val, fin_val, tar_val, fit_value

    # Reset global variables
    init_val = {}
    fin_val = {}
    tar_val = target_meta_values
    fit_value = float("inf")

    # Prepare data
    if isinstance(source_data, pd.DataFrame):
        column_names = source_data.columns
        source_np = source_data.values
    else:
        source_np = source_data
        column_names = None

    # Get data dimensions
    n_samples, n_features = source_np.shape

    # Identify categorical and continuous features
    continuous_idx = list(range(n_features - 1))
    categorical_idx = [n_features - 1]

    # Prepare categorical probabilities
    cat_probs = []
    if categorical_idx:
        for cat_idx in categorical_idx:
            unique_vals, counts = np.unique(source_np[:, cat_idx], return_counts=True)
            probs = counts / counts.sum()
            cat_probs.append(probs)

    # Train GMM for continuous features if needed
    gmm = None
    if mutation_type == "row_dist":
        try:
            gmm = GaussianMixture(n_components=min(3, len(source_np))).fit(
                source_np[:, continuous_idx]
            )
        except Exception as e:
            print(f"Error training GMM model: {e}")

    def create_individual():
        """Create an individual by copying the source data with some noise"""
        individual_data = source_np.copy()
        if forest_diffusion is not None:
            individual_data_numpy = forest_diffusion.generate(
                batch_size=len(individual_data)
            )
            individual_data_numpy = np.nan_to_num(individual_data_numpy)
            return creator.Individual(individual_data_numpy.flatten().tolist())
        return creator.Individual(individual_data.flatten().tolist())

    # Setup toolbox
    toolbox = base.Toolbox()
    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Register evaluation function
    toolbox.register(
        "evaluate",
        fitness_function_multi,
        meta_features=meta_features,
        target_values=target_meta_values,
        n_features=n_features,
        initial_data=source_np,
    )

    # Register selection operator (NSGA-III)
    ref_points = uniform_reference_points(nobj=2, p=12)
    toolbox.register("select", tools.selNSGA3WithMemory(ref_points))

    # Register crossover operator
    toolbox.register(
        "mate",
        crossover,
        cxpb=crossover_prob,
        row_mode_prob=row_mode_prob,
        n_features=n_features,
    )

    # Register mutation operator based on type
    mutation_params = {
        "mutation_prob": mutation_prob,
        "categorical_idx": categorical_idx,
        "continuous_idx": continuous_idx,
        "cat_probs": cat_probs,
        "n_features": n_features,
    }

    if mutation_type == "all":
        mutation_params["gmm"] = gmm
        toolbox.register("mutate", mutate_all, **mutation_params)
    elif mutation_type == "row_dist" and gmm is not None:
        mutation_params["gmm"] = gmm
        toolbox.register("mutate", mutate_dist, **mutation_params)
    elif mutation_type == "row_noise":
        toolbox.register("mutate", mutate_noise, **mutation_params)
    elif mutation_type == "row_cov":
        toolbox.register("mutate", mutate_cov, **mutation_params)
    else:
        print(f"Unknown mutation type: {mutation_type}, using row_noise as default")
        toolbox.register("mutate", mutate_noise, **mutation_params)

    # Initialize population and statistics
    population = toolbox.population(n=population_size)
    pareto_front = tools.ParetoFront()

    # Setup statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min, axis=0)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)

    try:
        population, logbook = algorithms.eaMuPlusLambda(
            population,
            toolbox,
            mu=population_size,
            lambda_=population_size,
            cxpb=crossover_prob,
            mutpb=mutation_prob,
            ngen=generations,
            stats=stats,
            halloffame=pareto_front,
            verbose=True,
        )
    except Exception as e:
        print(f"Error in evolutionary algorithm: {e}")
        return source_np, None, source_np, None

    # Analyze Pareto front
    pareto_front = tools.ParetoFront()

    # Select final solution
    best_individual = select_solution_from_pareto(pareto_front)
    if best_individual is None:
        best_individual = tools.selBest(population, 1)[0]

    # Convert to numpy array and reshape
    synthetic_data = np.array(best_individual).reshape(n_samples, n_features)
    synthetic_data = np.nan_to_num(synthetic_data)

    # Calculate final meta-feature values
    meta_values = {
        "initial": {mf: compute_meta_feature(source_np, mf) for mf in meta_features},
        "target": target_meta_values,
        "final": {mf: compute_meta_feature(synthetic_data, mf) for mf in meta_features},
    }

    if column_names is not None:
        synthetic_data = pd.DataFrame(synthetic_data, columns=column_names)

    return synthetic_data, logbook, source_np, meta_values


def run_shift_convergence_experiment(
    shift_type,
    meta_features,
    mutation_type,
    output_dir="synthetic_data",
    n_samples=500,
    generations=100,
    source_file=None,
    target_file=None,
):
    """
    Run experiments comparing convergence of meta-features on datasets from real source and target data.
    """
    # Create output directory
    experiment_dir = os.path.join(output_dir, f"shift_{shift_type}")
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    if source_file and target_file:
        # Load from provided file paths
        source_df = pd.read_csv(source_file)
        source_df = source_df.iloc[:, 1:]
        print(source_df.head())
        target_df = pd.read_csv(target_file)
        target_df = target_df.iloc[:, 1:]
        print(target_df.head())
        print(f"Loaded data from {source_file} and {target_file}")

        # Identify categorical and numerical columns
        cat_indexes = source_df.select_dtypes(include=["object"]).columns.tolist()
        int_indexes = source_df.select_dtypes(include=["int64"]).columns.tolist()
        bin_indexes = source_df.select_dtypes(include=["bool"]).columns.tolist()

        # Convert to column indices
        cat_indexes = [source_df.columns.get_loc(col) for col in cat_indexes]
        int_indexes = [source_df.columns.get_loc(col) for col in int_indexes]
        bin_indexes = [source_df.columns.get_loc(col) for col in bin_indexes]

        # Initialize ForestDiffusion
        source_df_numpy = source_df.to_numpy()
        forest_diffusion = ForestDiffusionModel(
            source_df_numpy,
            n_t=2,
            duplicate_K=3,
            bin_indexes=bin_indexes,
            cat_indexes=cat_indexes,
            int_indexes=int_indexes,
            diffusion_type="flow",
            n_jobs=-1,
        )

    # Handle any data preprocessing required
    source_df.dropna(inplace=True)
    target_df.dropna(inplace=True)
    source_df.reset_index(drop=True, inplace=True)
    target_df.reset_index(drop=True, inplace=True)

    # Compute target meta-feature values
    target_values = {}
    for meta_feature in meta_features:
        target_val = compute_meta_feature(target_df, meta_feature)
        if target_val is not None:
            target_values[meta_feature] = target_val

    # Prepare summary results
    results = {"shift_type": shift_type, "experiments": []}

    # Save source and target datasets (keeping only synthetic datasets)

    # Create a DataFrame to store best results for the mutation type
    best_results = pd.DataFrame(columns=["mutation_type"] + meta_features)

    # Run experiment for the mutation type

    # Create directory for this mutation type
    mutation_dir = os.path.join(experiment_dir, mutation_type)
    if not os.path.exists(mutation_dir):
        os.makedirs(mutation_dir)

    # Run single experiment
    print(f"Running experiment with {mutation_type} mutation")

    try:
        # Generate synthetic data
        synthetic_data, logbook, original_data, meta_values = generate_synthetic_data(
            mutation_type=mutation_type,
            source_data=source_df,
            meta_features=meta_features,
            target_meta_values=target_values,
            population_size=2,
            generations=generations,
            forest_diffusion=forest_diffusion,
        )

        if logbook is not None:
            # Calculate total error across all meta-features
            total_error = sum(
                np.linalg.norm(meta_values["target"][mf] - meta_values["final"][mf])
                for mf in meta_features
            )

            best_synthetic_data = (
                synthetic_data.copy()
                if isinstance(synthetic_data, pd.DataFrame)
                else pd.DataFrame(synthetic_data, columns=source_df.columns)
            )

            # Plot convergence
            convergence_plot = plot_convergence_multi(
                mutation_dir, logbook, meta_features, mutation_type, meta_values
            )

            # Create pairplot comparing source, target and synthetic data
            plt.figure(figsize=(15, 15))

            # Combine data for plotting
            source_data = source_df.copy()
            target_data = target_df.copy()
            synthetic_data_df = pd.DataFrame(synthetic_data, columns=source_df.columns)

            # Add dataset labels
            source_data["Dataset"] = "Source"
            target_data["Dataset"] = "Target"
            synthetic_data_df["Dataset"] = "Synthetic"

            # Combine all data
            combined_data = pd.concat([source_data, target_data, synthetic_data_df])

            # Create pairplot
            g = sns.pairplot(combined_data, hue="Dataset", diag_kind="kde")
            g.fig.suptitle(f"Feature Distribution Comparison - {mutation_type}", y=1.02)
            pairplot_path = os.path.join(mutation_dir, f"pairplot_{mutation_type}.png")
            plt.savefig(pairplot_path, bbox_inches="tight", dpi=150)
            plt.close()

            # Store experiment result
            experiment_result = {
                "mutation_type": mutation_type,
                "initial_value": meta_values["initial"],
                "target_value": meta_values["target"],
                "final_value": meta_values["final"],
                "absolute_error": {
                    mf: np.linalg.norm(
                        meta_values["target"][mf] - meta_values["final"][mf]
                    )
                    for mf in meta_features
                },
                "relative_error": {
                    mf: (
                        np.linalg.norm(
                            meta_values["target"][mf] - meta_values["final"][mf]
                        )
                        / np.linalg.norm(meta_values["target"][mf])
                        if np.linalg.norm(meta_values["target"][mf]) != 0
                        else float("inf")
                    )
                    for mf in meta_features
                },
                "convergence_plot": convergence_plot,
                "pairplot": pairplot_path,
            }

            synthetic_data_path = os.path.join(
                mutation_dir, f"synthetic_data_{mutation_type}.csv"
            )
            synthetic_data_df = (
                synthetic_data.copy()
                if isinstance(synthetic_data, pd.DataFrame)
                else pd.DataFrame(synthetic_data, columns=source_df.columns)
            )
            synthetic_data_df.to_csv(synthetic_data_path, index=False)
            print(f"Saved synthetic data for {mutation_type}")

    except Exception as e:
        print(f"Error in experiment: {e}")
        import traceback

        traceback.print_exc()
        return None

    # Save the synthetic dataset for this mutation type
    if best_synthetic_data is not None:
        best_synthetic_data.to_csv(
            os.path.join(mutation_dir, f"synthetic_data_{mutation_type}.csv"),
            index=False,
        )
        print(f"Saved synthetic dataset for {mutation_type}")

    # Add results to best_results DataFrame
    if "experiment_result" in locals():
        results["experiments"].append(experiment_result)

        # Add results to best_results DataFrame
        best_results = pd.concat(
            [
                best_results,
                pd.DataFrame(
                    {
                        "mutation_type": [mutation_type],
                        **{
                            mf: [experiment_result["absolute_error"][mf]]
                            for mf in meta_features
                        },
                    }
                ),
            ],
            ignore_index=True,
        )

    print("\nExperiment results:")
    print(best_results)

    return results


def plot_convergence_multi(
    output_dir, logbook, meta_features, mutation_type, meta_values
):
    """
    Plot convergence for multi-objective optimization - one plot per meta-feature showing Euclidean distance.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Extract data from logbook
    gen = logbook.select("gen")
    min_values = np.array(logbook.select("min"))
    avg_values = np.array(logbook.select("avg"))
    std_values = np.array(logbook.select("std"))

    # Create one plot per meta-feature
    plt.figure(figsize=(12, 6 * len(meta_features)))

    for i, meta_feature in enumerate(meta_features):
        plt.subplot(len(meta_features), 1, i + 1)

        # Plot min and average values
        plt.plot(gen, min_values[:, i], "r-", label=f"Best Distance", linewidth=2)
        plt.plot(gen, avg_values[:, i], "b--", label=f"Average Distance", linewidth=2)

        # Add error bands
        if std_values.size > 0:
            plt.fill_between(
                gen,
                avg_values[:, i] - std_values[:, i],
                avg_values[:, i] + std_values[:, i],
                alpha=0.2,
                color="b",
                label="±1 std",
            )

        plt.xlabel("Generation", fontsize=12)
        plt.ylabel(f"Euclidean Distance", fontsize=12)
        plt.title(
            f"{meta_feature} Convergence\nInitial Distance: {min_values[0, i]:.4f}, Final Distance: {min_values[-1, i]:.4f}",
            fontsize=14,
        )
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12, loc="upper right")

    plt.suptitle(
        f"Optimization Convergence with {mutation_type} Mutation Strategy",
        fontsize=16,
        y=1.02,
    )
    plt.tight_layout()

    # Save the figure
    filepath = os.path.join(output_dir, f"convergence_{mutation_type}.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()

    return filepath


if __name__ == "__main__":
    import os
    import glob
    from sklearn.preprocessing import LabelEncoder

    # Define output directory for experiment results
    OUTPUT_DIR = "synthetic_data"
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Define meta-features to analyze
    META_FEATURES = [
        "class_conc",
        "mut_inf",
    ]  # to add/reduce aims, also modify amount ref_points, weights
    # Define mutation type to test
    MUTATION_TYPE = "all"

    all_results = {}

    # Run experiment for electricity dataset
    dataset_name = input("Enter dataset name: ")
    source_file = input("Enter source dataset path: ")
    target_file = input("Enter target dataset path: ")

    if os.path.exists(target_file):
        print(f"\n===== Running experiments for {dataset_name} dataset =====\n")

        # Create a directory for this dataset
        dataset_dir = os.path.join(OUTPUT_DIR, dataset_name)
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)

        source_df = pd.read_csv(source_file)
        source_df = source_df.iloc[:, 1:]
        target_df = pd.read_csv(target_file)
        target_df = target_df.iloc[:, 1:]
        print(source_df.head())
        print(target_df.head())
        dataset_length = len(source_df)

        try:
            results = run_shift_convergence_experiment(
                shift_type=f"file:{dataset_name}",
                meta_features=META_FEATURES,
                mutation_type=MUTATION_TYPE,
                output_dir=dataset_dir,
                n_samples=dataset_length,
                generations=2,
                source_file=source_file,
                target_file=target_file,
            )

            all_results[dataset_name] = results

        except Exception as e:
            print(f"Error running experiment for {dataset_name}: {e}")
            import traceback

            traceback.print_exc()

    print("\n===== All experiments completed =====")
    print(f"Results saved to {OUTPUT_DIR}")
