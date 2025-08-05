import numpy as np
import random
from scipy.stats import kurtosis
from sklearn.feature_selection import mutual_info_classif

def compute_meta_features(X, y):
    """Compute meta-features for a dataset subset."""
    means = np.mean(X, axis=0)
    kurtoses = kurtosis(X, axis=0, nan_policy='raise')
    mi = mutual_info_classif(X, y, discrete_features=False, random_state=0)
    return np.concatenate([mi])

def create_individual_by_ordering(X, k):
    """Create an individual based on feature ordering patterns."""
    n, d = X.shape
    j = np.random.randint(d)
    sorted_indices = np.argsort(X[:, j])
    pattern = random.choice(['first_k', 'last_k', 'contiguous', 'every_other', 'random'])
    
    if pattern == 'first_k':
        test_indices = sorted_indices[:k]
    elif pattern == 'last_k':
        test_indices = sorted_indices[-k:]
    elif pattern == 'contiguous':
        start = random.randint(0, n - k)
        test_indices = sorted_indices[start:start + k]
    elif pattern == 'every_other': 
        step = 2
        start = random.randint(0, 1)
        candidate = sorted_indices[start::step]
        test_indices = candidate[:k] if len(candidate) >= k else sorted_indices[:k]
    else:
        test_indices = random.sample(list(sorted_indices), k)
    
    individual = np.zeros(n, dtype=int)
    individual[test_indices] = 1
    return individual.tolist()

def repair(individual, k):
    """Ensure exactly k test samples in the individual."""
    n = len(individual)
    current_ones = sum(individual)
    if current_ones == k:
        return individual
    
    indices = np.arange(n)
    if current_ones > k:
        ones_indices = [i for i, bit in enumerate(individual) if bit == 1]
        flip_indices = np.random.choice(ones_indices, current_ones - k, replace=False)
        for idx in flip_indices:
            individual[idx] = 0
    else:
        zeros_indices = [i for i, bit in enumerate(individual) if bit == 0]
        flip_indices = np.random.choice(zeros_indices, k - current_ones, replace=False)
        for idx in flip_indices:
            individual[idx] = 1
    return individual

def mutate(individual):
    """Swap a train and test sample to maintain test set size."""
    n = len(individual)
    zeros = [i for i, bit in enumerate(individual) if bit == 0]
    ones = [i for i, bit in enumerate(individual) if bit == 1]
    if not zeros or not ones:
        return individual
    idx0 = np.random.choice(zeros)
    idx1 = np.random.choice(ones)
    individual[idx0], individual[idx1] = 1, 0
    return individual

def maximize_distance_evolutionary(X, y, test_size=0.2, pop_size=50, n_generations=20, 
                                   mutation_rate=0.1, tournament_size=3, n_precomputation_splits=100, 
                                   random_state=None):
    """Evolutionary algorithm to maximize meta-feature distance between train/test splits."""
    if random_state is not None:
        np.random.seed(random_state)
        random.seed(random_state)
    
    n, d = X.shape
    k = int(round(test_size * n))
    
    # Precompute scaling factors using random splits
    all_meta = []
    indices = np.arange(n)
    for _ in range(n_precomputation_splits):
        np.random.shuffle(indices)
        train_idx = indices[k:]
        test_idx = indices[:k]
        try:
            meta_train = compute_meta_features(X[train_idx], y[train_idx])
            meta_test = compute_meta_features(X[test_idx], y[test_idx])
            all_meta.append(meta_train)
            all_meta.append(meta_test)
        except Exception:
            continue
    
    if not all_meta:
        stds = np.ones(compute_meta_features(X, y).shape[0])
    else:
        all_meta_arr = np.array(all_meta)
        stds = np.std(all_meta_arr, axis=0) + 1e-9  # Avoid division by zero
    
    # Initialize population using feature ordering patterns
    population = []
    for _ in range(pop_size):
        ind = create_individual_by_ordering(X, k)
        population.append(ind)
    
    best_individual = None
    best_fitness = -np.inf
    fiitness_all = []
    # Evolutionary loop
    for gen in range(n_generations):
        fitness = []
        for ind in population:
            train_indices = [i for i, bit in enumerate(ind) if bit == 0]
            test_indices = [i for i, bit in enumerate(ind) if bit == 1]

            meta_train = compute_meta_features(X[train_indices], y[train_indices])
            meta_test = compute_meta_features(X[test_indices], y[test_indices])
            diff = meta_train - meta_test
            normalized_diff = diff
            dist = np.linalg.norm(normalized_diff)
            fitness.append(dist)
            if dist > best_fitness:
                best_fitness = dist
                best_individual = ind.copy()
        print(best_fitness)
        fiitness_all.append(best_fitness)

        # Tournament selection
        parents = []
        for _ in range(pop_size):
            tournament_indices = random.sample(range(pop_size), tournament_size)
            tournament_fitness = [fitness[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            parents.append(population[winner_idx])
        
        # Crossover and mutation
        offspring = []
        for i in range(0, pop_size, 2):
            p1, p2 = parents[i], parents[i+1]
            c1, c2 = [], []
            for j in range(n):
                if random.random() < 0.5:
                    c1.append(p1[j])
                    c2.append(p2[j])
                else:
                    c1.append(p2[j])
                    c2.append(p1[j])
            c1 = repair(c1, k)
            c2 = repair(c2, k)
            if random.random() < mutation_rate:
                c1 = mutate(c1)
            if random.random() < mutation_rate:
                c2 = mutate(c2)
            offspring.extend([c1, c2])
        
        # Elitism
        if best_individual is not None:
            offspring[0] = best_individual
        
        population = offspring
    
    # Extract best solution
    if best_individual is None:
        train_indices = list(range(n - k))
        test_indices = list(range(n - k, n))
    else:
        train_indices = [i for i, bit in enumerate(best_individual) if bit == 0]
        test_indices = [i for i, bit in enumerate(best_individual) if bit == 1]
    
    return train_indices, test_indices, fiitness_all