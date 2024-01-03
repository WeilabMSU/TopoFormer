""" Summary: python code for the normal simplicial complex Laplacian

    Author:
        Dong Chen
    Create:
        2023-04-07
    Modify:
        2023-04-07
    Dependencies:
        python                    3.7.4
        numpy                     1.21.5
"""


import numpy as np
import itertools
from functools import wraps
import copy
import argparse
import sys
import time
from scipy.spatial import distance


def timeit(func):
    """ Timer """
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        # print(f"{'='*5} Function {func.__name__}{args} {kwargs} Took {total_time:.3f} seconds {'='*5}")
        print(f"{'='*5} Function - {func.__name__} - took {total_time:.3f} seconds {'='*5}")
        return result
    return timeit_wrapper


class statistic_eigvalues(object):
    '''Input is 1-D array'''
    def __init__(self, eigvalues: np.array) -> None:
        digital = 5
        values = np.round(eigvalues, 5)
        self.all_values = sorted(values)
        self.nonzero_values = values[np.nonzero(values)]
        self.count_zero = len(values) - np.count_nonzero(values)
        self.max = np.max(values)
        self.sum = np.round(np.sum(values), digital)

        if len(self.nonzero_values) > 0:
            self.nonzero_mean = np.round(np.mean(self.nonzero_values), digital)
            self.nonzero_std = np.round(np.std(self.nonzero_values), digital)
            self.nonzero_min = np.round(np.min(self.nonzero_values), digital)
            self.nonzero_var = np.round(np.var(self.nonzero_values), digital)
        else:
            # if none nonzero min, set it as 0
            self.nonzero_mean = 0
            self.nonzero_std = 0
            self.nonzero_min = 0
            self.nonzero_var = 0


class SimplicialComplexLaplacian(object):
    def __init__(self, eigenvalue_method='numpy_eigvalsh'):
        self.distance_matrix = None
        if eigenvalue_method == 'numpy_eigvalsh':
            self.eigvalue_calculator = np.linalg.eigvalsh

    def utils_powersets(self, nodes: list, max_dim: int = 2) -> dict:
        complete_edge_dict = {i: [] for i in range(max_dim)}

        max_len = min([len(nodes), max_dim])
        for i in range(max_len+1):
            complete_edge_dict[i] = list(itertools.combinations(nodes, i+1))
        return complete_edge_dict

    def adjacency_map_to_simplex(self, adjacency_matrix: np.array, max_dim: int = 1) -> dict:
        """
            Given an adjacency matrix A for an undirected graph,
            construct the clique complex of the graph.
        """
        n = adjacency_matrix.shape[0]  # Number of nodes in the graph
        simplicial_complex = {dim: [] for dim in range(max_dim+1)}  # List of simplices in the clique complex

        # Add the 0-simplices (nodes)
        simplicial_complex[0] = [(i, ) for i in range(n)]

        # Add higher-dimensional simplices corresponding to cliques of size > 1
        target_dim = min(max_dim, n)
        for k in range(1, target_dim+1):
            for S in itertools.combinations(range(n), k+1):
                if all(adjacency_matrix[i,j] for i in S for j in S if i < j):
                    simplicial_complex[k].append(tuple(S))
        return simplicial_complex
    
    def complex_to_boundary_matrix(self, complex: dict) -> dict:

        # For dim_0, boundary matrix shape [len(node), 1]
        boundary_matrix_dict = {dim_n: None for dim_n in complex.keys()}
        for dim_n in sorted(complex.keys()):
            if dim_n == 0:
                boundary_matrix_dict[dim_n] = np.zeros([len(complex[0]), ])
                continue
            simplex_dim_n = sorted(complex[dim_n])
            simplex_dim_n_minus_1 = sorted(complex[dim_n-1])
            if len(simplex_dim_n) == 0:
                break
            boundary_matrix_dict[dim_n] = np.zeros(
                [len(simplex_dim_n_minus_1), len(simplex_dim_n)])
            for idx_n, simplex_n in enumerate(simplex_dim_n):
                for omitted_n in range(len(simplex_n)):
                    omitted_simplex = tuple(np.delete(simplex_n, omitted_n))
                    omitted_simplex_idx = simplex_dim_n_minus_1.index(omitted_simplex)
                    boundary_matrix_dict[dim_n][omitted_simplex_idx, idx_n] = (-1)**omitted_n

        self.has_boundary_max_dim = dim_n
        return boundary_matrix_dict

    def boundary_to_laplacian_matrix(self, boundary_matrix_dict: dict) -> dict:
        laplacian_matrix_dict = {}
        for dim_n in sorted(boundary_matrix_dict.keys()):
            boundary_matrix = boundary_matrix_dict[dim_n]
            if dim_n >= self.has_boundary_max_dim:
                break
            elif dim_n == 0 and boundary_matrix_dict[dim_n+1] is not None:
                laplacian_matrix_dict[dim_n] = np.dot(boundary_matrix_dict[dim_n+1], boundary_matrix_dict[dim_n+1].T)
            elif dim_n == 0 and boundary_matrix_dict[dim_n+1] is None:
                laplacian_matrix_dict[dim_n] = np.zeros([len(boundary_matrix_dict[0])]*2)
            elif dim_n > 0 and boundary_matrix_dict[dim_n+1] is None:
                laplacian_matrix_dict[dim_n] = np.dot(boundary_matrix.T, boundary_matrix)
                break
            else:
                laplacian_matrix_dict[dim_n] = np.dot(
                    boundary_matrix_dict[dim_n+1], boundary_matrix_dict[dim_n+1].T) + np.dot(boundary_matrix.T, boundary_matrix)
        return laplacian_matrix_dict

    def simplicialComplex_laplacian_from_connected_mat(self, adjacency_matrix: np.array, max_dim: int = 1) -> np.array:
        self.max_dim = max_dim
        self.max_boundary_dim = max_dim + 1
        # node_idx_list = sorted(list(range(adjacency_matrix.shape[0])))
        complex = self.adjacency_map_to_simplex(adjacency_matrix, self.max_boundary_dim)
        boundary_matrix_dict = self.complex_to_boundary_matrix(complex)
        laplacian_matrix_dict = self.boundary_to_laplacian_matrix(boundary_matrix_dict)

        laplacian_eigenv = {}
        for dim_n in range(self.max_dim+1):
        # for dim_n, laplacian_matrix in laplacian_matrix_dict.items():
            if dim_n in laplacian_matrix_dict:
                laplacian_matrix = laplacian_matrix_dict[dim_n]
                eig_value = self.eigvalue_calculator(laplacian_matrix)
                eig_value = eig_value.real
                laplacian_eigenv[dim_n] = sorted(np.round(eig_value, 5))
            else:
                laplacian_eigenv[dim_n] = None
        self.laplacian_matrix_dict = laplacian_matrix_dict
        return laplacian_eigenv

    def persistent_simplicialComplex_laplacian(
        self, input_data: np.array = None,
        max_adjacency_matrix: np.array = None, min_adjacency_matrix: np.array = None,
        is_distance_matrix: bool = False, max_dim: int = 1, filtration: np.array = None,
        cutoff_distance: float = None, step_dis: float = None, print_by_step: bool = True,
    ) -> np.array:
        # the default data is cloudpoints
        if is_distance_matrix:
            distance_matrix = input_data
            points_num = distance_matrix.shape[0]
        else:
            cloudpoints = input_data
            points_num = cloudpoints.shape[0]
            distance_matrix = distance.cdist(cloudpoints, cloudpoints)

        if max_adjacency_matrix is None:
            max_adjacency_matrix = np.ones([points_num, points_num], dtype=int)
            np.fill_diagonal(max_adjacency_matrix, 0)
        
        if min_adjacency_matrix is None:
            min_adjacency_matrix = np.zeros([points_num, points_num], dtype=int)

        if filtration is None:
            filtration = np.arange(0, cutoff_distance, step_dis)
        
        all_laplacian_features = []
        adjacency_matrix_temp = np.ones([points_num]*2, dtype=int)
        for threshold_dis in filtration:
            adjacency_matrix = (((distance_matrix <= threshold_dis) * max_adjacency_matrix + min_adjacency_matrix) > 0)

            if not (adjacency_matrix == adjacency_matrix_temp).all():
                adjacency_matrix_temp = copy.deepcopy(adjacency_matrix)
                laplacian_eigenv = self.simplicialComplex_laplacian_from_connected_mat(adjacency_matrix, max_dim)
                all_laplacian_features.append(laplacian_eigenv)
            else:
                all_laplacian_features.append(all_laplacian_features[-1])
                
            if print_by_step:
                for dim_ii in range(max_dim):
                    print(f"filtration param: {threshold_dis} dim_n: {dim_ii} eigenvalues:{laplacian_eigenv[dim_ii]}")

        return all_laplacian_features

    def persistent_simplicialComplex_laplacian_dim0(
        self, input_data: np.array = None,
        max_adjacency_matrix: np.array = None, min_adjacency_matrix: np.array = None,
        is_distance_matrix: bool = False, max_dim: int = 1, filtration: np.array = None,
        cutoff_distance: float = None, step_dis: float = None, print_by_step: bool = True,
    ) -> np.array:
        # the default data is cloudpoints
        if is_distance_matrix:
            distance_matrix = input_data
            points_num = distance_matrix.shape[0]
        else:
            cloudpoints = input_data
            points_num = cloudpoints.shape[0]
            distance_matrix = distance.cdist(cloudpoints, cloudpoints)

        if max_adjacency_matrix is None:
            max_adjacency_matrix = np.ones([points_num, points_num], dtype=int)
            np.fill_diagonal(max_adjacency_matrix, 0)
        
        if min_adjacency_matrix is None:
            min_adjacency_matrix = np.zeros([points_num, points_num], dtype=int)

        if filtration is None:
            filtration = np.arange(0, cutoff_distance, step_dis)
        
        all_laplacian_features = []
        adjacency_matrix_temp = np.ones([points_num]*2, dtype=int)
        for threshold_dis in filtration:
            adjacency_matrix = (((distance_matrix <= threshold_dis) * max_adjacency_matrix + min_adjacency_matrix) > 0)

            if not (adjacency_matrix == adjacency_matrix_temp).all():
                adjacency_matrix_temp = copy.deepcopy(adjacency_matrix)
                laplacian_matrix_dim0 = np.diag(np.sum(adjacency_matrix, axis=0)) - adjacency_matrix
                eig_value = self.eigvalue_calculator(laplacian_matrix_dim0)
                eig_value = eig_value.real
                laplacian_eigenv = {0: sorted(np.round(eig_value, 5))}
                all_laplacian_features.append(laplacian_eigenv)
            else:
                all_laplacian_features.append(all_laplacian_features[-1])
                
            if print_by_step:
                for dim_ii in range(max_dim):
                    print(f"filtration param: {threshold_dis} dim_n: {dim_ii} eigenvalues:{laplacian_eigenv[dim_ii]}")

        return all_laplacian_features


def main():
    aa = SimplicialComplexLaplacian()
    adjacency_matrix = np.array([
        [0, 1, 0, 1, 0],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [1, 0, 1, 0, 1],
        [0, 0, 0, 1, 0],
    ])
    adjacency_matrix = np.array([
        [0, 1, 1, 0],
        [1, 0, 1, 0],
        [1, 1, 0, 1],
        [0, 0, 1, 0],
    ])
    # adjacency_matrix = np.array([
    #     [0, 1, 0, 0, 1, 0],
    #     [0, 0, 1, 0, 0, 0],
    #     [0, 0, 0, 1, 0, 0],
    #     [0, 0, 0, 0, 0, 0],
    #     [0, 0, 1, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0],
    # ])
    ww = aa.adjacency_map_to_simplex(adjacency_matrix, max_dim=2)
    print(ww)
    feat = aa.simplicialComplex_laplacian_from_connected_mat(adjacency_matrix, max_dim=2)
    print(feat)
    print(aa.laplacian_matrix_dict[0])

    print(np.diag(np.sum(adjacency_matrix, axis=0)) - adjacency_matrix)
    return None


if __name__ == "__main__":
    main()
