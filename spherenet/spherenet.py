#!/usr/bin/env python3

import numpy as np
from numba_progress import ProgressBar
import numba as nb
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.neighbors import LocalOutlierFactor
import pickle
from textwrap import dedent

# GLOBAL VARS
metrics_available = ['std', 'minkowski', 'euclid', 'hamming', 'max', 'cosine', 'jaccard', 'dice', 'canberra', 'braycurtis', 'correlation', 'yule', 'havensine', 'sum', 'mse', 'ots']
pred_modes = ['conservative', 'careful', 'force']


# CLASSIFICATION CLASSES
class SphereNet:
    """
    SphereNet for one inside classification
    """

    
    def __init__(self, in_class=1, min_dist_scaler=1.0, min_radius_threshold=-1, optimization_tolerance=0, optimization_repetitions=1, optimization_parallel=False, min_num_classified=2, max_spheres_used=-1, metric='euclid', p=2, standard_scaling=False, normalization=False, remove_training_outliers=True, verbosity=0):
        """
        Initialize the SphereNet model
        
        :param in_class: the class that will be classified as True or inside the spheres
        :param min_dist_scaler: The minimum distance between spheres
        :param min_radius_threshold: The minimum radius of a sphere (if is -1, there is none)
        :param optimization_tolerance: The optimization tolerance (higher = more optimization but less performance)
        :param optimization_repetitions: The number the optimizer will be let rerun. The first run(s) some optimizations can be missed when let run parallel. When set to 0, optimization will not be performed.
        :param optimization_parallel: Parallel optimization does not optimize optimally because of race coniditons, but is much faster. When using parallel optimization, optimization_repetitions should be greater than 1.
        :param min_num_classified: A sphere must at least classifiy this amount of points to be kept. Can be used for outlier removal.
        :param max_spheres_used: The maximum number of spheres used for classification (-1 = no limit)
        :param metric: The distance metric to be used. Available: ['minkowski', 'hamming', 'max', 'min']
        :param p: the p hyperparam (order of magnitude) when using the minkowski distance (p=1 is manhattan, p=2 is euclid)
        :param standard_scaling: if should use standard scaling. Can also be a StandardScaler object
        :param normalization: if should use normalization. Can also be a Normalizer object
        :param remove_training_outliers: if should find outliers in training data
        :param verbosity: The verbosity level (between 0 and 2)
        """
        
        # assign args to self
        self.in_class = in_class
        self.min_dist_scaler = min_dist_scaler
        self.min_radius_threshold = min_radius_threshold
        self.optimization_tolerance = optimization_tolerance
        self.optimization_repetitions = optimization_repetitions
        self.optimization_parallel = optimization_parallel
        self.min_num_classified = min_num_classified
        self.max_spheres_used = max_spheres_used
        self.verbosity = verbosity
        self.standard_scaling = bool(standard_scaling)  # bools because can also be the scaler objects
        self.normalization = bool(normalization)  # see above
        self.remove_training_outliers = remove_training_outliers
        self.metric = metric
        self.p = p
        # set metric index
        self._metric = metrics_available.index(metric)
 
        # init scaler(s), if classes passed, set those as scalers
        if standard_scaling:
            if isinstance(standard_scaling, StandardScaler):
                self.standard_scaler = standard_scaling
            else:
                self.standard_scaler = StandardScaler()

        if normalization:
            if isinstance(normalization, Normalizer):
                self.normalizer = normalization
            else:
                self.normalizer = Normalizer()
    
 
    @staticmethod
    @nb.njit(parallel=True, fastmath=True)
    def _calculate_spheres(X_IN, X_OUT, progress, min_dist_scaler, min_radius_threshold, min_num_classified, _metric, p):
        """
        determine n spheres center and their single sphere performance
        :param X_IN: the inside points
        :param X_OUT: the outside points
        :param progress: the progress bar
        :return: the performance, performance length, radii and centers for point packages
        """
                
        length_in, length_out = X_IN.shape[0], X_OUT.shape[0]
        # create arrays to be returned
        perf = np.full((length_in, length_in), False)
        radii = np.zeros(length_in, dtype=np.float64)
        perf_len = np.zeros(length_in)

        # rows to be used (not filtered by threshold)
        rows = np.full(length_in, True, dtype=np.bool_)
        for i in nb.prange(length_in):
            # update progress
            if progress is not None:
                progress.update(1)
                
            # temp dist array (inside loop for evading race condition in parallelity)
            dist = np.zeros(length_out, dtype=np.float64)

            # calculate the distance to the nearest outer point
            # since is 2d array perform on loop
            for j in range(length_out):
                # WHY THIS MESSY CODE (UGLY WORKAROUND) AND NOT JUST A METRIC FUNCTION POINTER?
                # the numba compiler seems to not be able to optimize
                # good enough when using function pointers
                # At least I had no success with it.
                if _metric == 0:
                    dist[j] = np.std(X_IN[i] - X_OUT[j])  # std
                elif _metric == 1:
                    dist[j] = np.sum((X_IN[i] - X_OUT[j]) ** p) ** (1. / p) # minkowski
                elif _metric == 2:
                    dist[j] = np.sqrt(np.sum((X_IN[i] - X_OUT[j]) ** 2.))  # euclid
                elif _metric == 3:
                    dist[j] = np.sum(np.abs(X_IN[i] - X_OUT[j])) / X_IN[i].shape[0]  # hamming
                elif _metric == 4:
                    dist[j] = np.max(np.abs(X_IN[i] - X_OUT[j]))  # max
                elif _metric == 5:
                    dist[j] = 1 - np.sum(X_IN[i] * X_OUT[j]) / (np.sqrt(np.sum(X_IN[i] ** 2)) * np.sqrt(np.sum(X_OUT[j] ** 2)) + 1e-8)  # cosine
                elif _metric == 6:
                    dist[j] = 1 - np.sum(np.minimum(X_IN[i], X_OUT[j])) / (np.sum(np.maximum(X_IN[i], X_OUT[j])) + 1e-8)  # jaccard
                elif _metric == 7:
                    dist[j] = 1 - 2 * np.sum(np.minimum(X_IN[i], X_OUT[j])) / (np.sum(X_IN[i]) + np.sum(X_OUT[j]) + 1e-8)  # dice
                elif _metric == 8: 
                    dist[j] = np.sum(np.abs(X_IN[i] - X_OUT[j]) / (np.abs(X_IN[i]) + np.abs(X_OUT[j]) + 1e-8))  # canberra
                elif _metric == 9:
                    dist[j] = np.sum(np.abs(X_IN[i] - X_OUT[j])) / (np.sum(np.abs(X_IN[i] + X_OUT[j])) + 1e-8)  # braycurtis
                elif _metric == 10:
                    dist[j] = 1 - np.corrcoef(X_IN[i], X_OUT[j])[0, 1]  # correlation
                elif _metric == 11:
                    dist[j] = np.sum(X_IN[i] * X_OUT[j]) / (np.sum(np.abs(X_IN[i] - X_OUT[j])) + 1e-8)  # yule
                elif _metric == 12:
                    dist[j] = np.arccos(np.sum(X_IN[i] * X_OUT[j]) / (np.sqrt(np.sum(X_IN[i] ** 2)) * np.sqrt(np.sum(X_OUT[j] ** 2)) + 1e-8)) / np.pi  # havensine
                elif _metric == 13:
                    dist[j] = np.sum(X_IN[i] - X_OUT[j])  # sum
                elif _metric == 14:
                    dist[j] = np.sum((X_IN[i] - X_OUT[j]) ** 2) / X_IN[i].shape[0]  # mse
                elif _metric == 15:
                    dist[j] = 1. / np.sum((X_IN[i] - X_OUT[j]) ** 2. + 1e-8)  # ots

            # scale and get min from dist array
            radius = min_dist_scaler * np.min(dist)

            # only add if radius is larger than threshold
            if min_radius_threshold == -1 or radius > min_radius_threshold:
                radii[i] = radius

                # calculate performance (how many of the IN points are classified as in by just this one N-sphere)
                # distances < radius of this sphere
                # since is 2d array perform on loop, set each index directly
                for j in range(length_in):
                    # WHY THIS MESSY CODE (UGLY WORKAROUND) AND NOT JUST A METRIC FUNCTION POINTER?
                    # the numba compiler seems to not be able to optimize
                    # good enough when using function pointers
                    # At least I had no success with it.
                    if _metric == 0:
                        perf[i, j] = np.std(X_IN[i] - X_IN[j]) < radius  # std
                    elif _metric == 1:
                        perf[i, j] = np.sum((X_IN[i] - X_IN[j]) ** p) ** (1. / p) < radius # minkowski
                    elif _metric == 2:
                        perf[i, j] = np.sqrt(np.sum((X_IN[i] - X_IN[j]) ** 2.)) < radius  # euclid
                    elif _metric == 3:
                        perf[i, j] = np.sum(np.abs(X_IN[i] - X_IN[j])) / X_IN[i].shape[0] < radius  # hamming
                    elif _metric == 4:
                        perf[i, j] = np.max(np.abs(X_IN[i] - X_IN[j])) < radius  # max
                    elif _metric == 5:
                        perf[i, j] = 1 - np.sum(X_IN[i] * X_IN[j]) / (np.sqrt(np.sum(X_IN[i] ** 2)) * np.sqrt(np.sum(X_IN[j] ** 2)) + 1e-8) < radius  # cosine
                    elif _metric == 6:
                        perf[i, j] = 1 - np.sum(np.minimum(X_IN[i], X_IN[j])) / (np.sum(np.maximum(X_IN[i], X_IN[j])) + 1e-8) < radius  # jaccard
                    elif _metric == 7:
                        perf[i, j] = 1 - 2 * np.sum(np.minimum(X_IN[i], X_IN[j])) / (np.sum(X_IN[i]) + np.sum(X_IN[j]) + 1e-8) < radius  # dice
                    elif _metric == 8: 
                        perf[i, j] = np.sum(np.abs(X_IN[i] - X_IN[j]) / (np.abs(X_IN[i]) + np.abs(X_IN[j]) + 1e-8)) < radius  # canberra
                    elif _metric == 9:
                        perf[i, j] = np.sum(np.abs(X_IN[i] - X_IN[j])) / (np.sum(np.abs(X_IN[i] + X_IN[j])) + 1e-8) < radius  # braycurtis
                    elif _metric == 10:
                        perf[i, j] = 1 - np.corrcoef(X_IN[i], X_IN[j])[0, 1] < radius  # correlation
                    elif _metric == 11:
                        perf[i, j] = np.sum(X_IN[i] * X_IN[j]) / (np.sum(np.abs(X_IN[i] - X_IN[j])) + 1e-8) < radius  # yule
                    elif _metric == 12:
                        perf[i, j] = np.arccos(np.sum(X_IN[i] * X_IN[j]) / (np.sqrt(np.sum(X_IN[i] ** 2)) * np.sqrt(np.sum(X_IN[j] ** 2)) + 1e-8)) / np.pi < radius  # havensine
                    elif _metric == 13:
                        perf[i, j] = np.sum(X_IN[i] - X_IN[j]) < radius  # sum
                    elif _metric == 14:
                        perf[i, j] = np.sum((X_IN[i] - X_IN[j]) ** 2) / X_IN[i].shape[0] < radius  # mse
                    elif _metric == 15:
                        perf[i, j] = 1. / np.sum((X_IN[i] - X_IN[j]) ** 2. + 1e-8) < radius  # ots
                 
                    
                # calculate the sum of correctly classified performances
                perf_len[i] = perf[i].sum()
                
                # if this sum is smaller than a given value, set this row to false
                if perf_len[i] < min_num_classified:
                    rows[i] = False
            else:
                rows[i] = False
                

        return perf[rows], perf_len[rows], radii[rows], X_IN[rows]  # cut rows; X_IN as centers
  

    
    @staticmethod 
    def _remove_ambiguity(perf, perf_len, radii, centers, progress, optimization_tolerance):
        """
        Remove ambiguity / optimize the spheres by removing overlapping sphere results.
        :param perf: the performance of the spheres
        :param perf_len: the number of correctly classified points for each row on training data
        :param radii: the radii of the spheres
        :param centers: the centers of the spheres
        :param progress: the progress bar
        :return: optimized performances, radii and centers that can be used for classification
        """
  
        # sort performances, radii and centers in the same manner
        # as needed by algorithm
        # sort:
        # small to large (stl)
        # and
        # large to small (lts)
        idx_stl = perf_len.argsort()
        idx_lts = idx_stl[::-1]
        perf_len_stl = perf_len[idx_stl]
        perf_len_lts = perf_len[idx_lts]
        perf_lts = perf[idx_lts]
        perf_stl = perf[idx_stl]
        radii_lts = radii[idx_lts]
        centers_lts = centers[idx_lts]

        rows_len, cols_len = perf_stl.shape
        # rows is sorted by lts, true means is disabled, false means enabled
        rows = np.full(
            rows_len, False, dtype=np.bool_
        )  

        # outer loop:
        # loop rows in stl sorted list
        # because we want to keep spheres classifying worse to be rather thrown out
        # than spheres better
        for stl_idx in nb.prange(rows_len):
            # update progress
            if progress is not None:
                progress.update(1)

            # get this rows lts index
            this_lts_idx = rows_len - 1 - stl_idx
            # disable this row for comparison in loop
            rows[this_lts_idx] = True

            # reset trues count and row_disabled bool etc.
            trues_found, trues_should = 0, 0
            tolerance_left = optimization_tolerance
            row_disabled = False

            # do a depth first search:
            # loop through columns first
            # then (nested) through rows
            # much less columns are activated than rows
            for col_idx in range(cols_len):
                # check if row is disabled already (everything needed has been found)
                # because then this loop can be directly broken
                if row_disabled:
                    break

                # only do a check if column has been
                # even classified correcly by current sphere (row)
                if perf_stl[stl_idx][col_idx]:
                    # this is activated, trues should is +1 because this one should be found
                    trues_should += 1

                    # continue with lts sorted loop
                    # it is more likely to find a True value
                    # if the rows with most true values are on top
                    for lts_idx in range(rows_len):
                        # skip row if is deactivated already
                        if rows[lts_idx]:
                            continue

                        # else if finds a True value in this column
                        # break the loop and mark one more true as found
                        if perf_lts[lts_idx][col_idx]:
                            trues_found += 1

                            # if the amount of trues needed for knowing
                            # that the sphere is amiguous has been found
                            # we can also break the outer loop
                            # and set the row as disabled
                            if perf_len_stl[stl_idx] == trues_found:
                                row_disabled = True

                            # anyways, the row (depth) loop can be stopped
                            break


                    # if is not found,
                    # use tolerance to have a bool found (easier removal)
                    # but if tolerance is used up, sphere has to be disabled
                    if trues_found != trues_should:
                        tolerance_left -= 1

                        # no tolerance anymore, diable row and break loop
                        if tolerance_left == -1:
                            break

                        trues_found += 1


            # set enabled,
            # if no amiguousity has been found
            # set disabled, if has been found
            rows[this_lts_idx] = row_disabled


        # reverse rows for indexing
        rows = ~rows

        # return perfomances, centers and radii used for classification
        return perf_lts[rows], perf_len_lts[rows], centers_lts[rows], radii_lts[rows]
 
    
    def fit(self, X, y):
        """
        Fit the SphereNet model to the data
        :param X: The input data
        :param y: The target data 
        :return: The fitted model (self)
        """

        if self.verbosity > 1:
            print(f'Fitting SphereNet model for in_class {self.in_class}')

        # fit scaler and transform X data in same step
        if self.standard_scaling:
            if self.verbosity > 1:
                print('fitting standard scaler')
                
            # if is not fitted yet, fit
            if not hasattr(self.standard_scaler, 'n_features_in_'):
                X = self.standard_scaler.fit_transform(X)
            
        if self.normalization:
            if self.verbosity > 1:
                print('transforming with normalizer')
                
            X = self.normalizer.transform(X)
         
        # remove outliers with LocalOutlierFactor from IN and OUT distributions
        # if enabled
        if self.remove_training_outliers:
            if self.verbosity > 1:
                print('removing outliers from IN and OUT training distributions')
            
            # array marking if is no outlier
            not_outliers = np.full(len(X), False)
            
            # for IN and OUT distribution let LocalOutlierFactor run
            cl_idx = y == self.in_class
            not_outliers[cl_idx] = LocalOutlierFactor().fit_predict(X[cl_idx]) == 1
            
            cl_idx = y != self.in_class
            not_outliers[cl_idx] = LocalOutlierFactor().fit_predict(X[cl_idx]) == 1
            
            # overwrite X
            X = X[not_outliers]
            y = y[not_outliers]
            
        
        # get in X_IN and X_OUT points
        X_IN = X[y == self.in_class]
        X_OUT = X[y != self.in_class]

        if self.verbosity > 1:
            print("X_IN.shape:", X_IN.shape)
            print("X_OUT.shape:", X_OUT.shape)
            
            print('Calculating radii and centers...')

        # calculate the spheres and their performance
        if self.verbosity > 0:
            with ProgressBar(total=X_IN.shape[0]) as progress:
                self._perf, self._perf_len, self.cl_radii, self.cl_centers = self._calculate_spheres(X_IN, X_OUT, progress, self.min_dist_scaler, self.min_radius_threshold, self.min_num_classified, self._metric, self.p)
        else:
            self._perf, self._perf_len, self.cl_radii, self.cl_centers = self._calculate_spheres(X_IN, X_OUT, None, self.min_dist_scaler, self.min_radius_threshold, self.min_num_classified, self._metric, self.p)

        if self.verbosity > 1: 
            print('Removing ambiguity...')

        # remove ambiguity / optimize the spheres
        # repeat as often as defined
        for rep in range(self.optimization_repetitions):
            if self.verbosity > 1:
                print(f'Repeating optimization: {rep}')
            
            if self.verbosity > 0:
                with ProgressBar(total=self._perf.shape[0]) as progress:
                    self._perf, self._perf_len, self.cl_centers, self.cl_radii = nb.njit(self._remove_ambiguity, parallel=self.optimization_parallel, fastmath=True)(self._perf, self._perf_len, self.cl_radii, self.cl_centers, progress, self.optimization_tolerance)
            else:
                self._perf, self._perf_len, self.cl_centers, self.cl_radii = nb.njit(self._remove_ambiguity, parallel=self.optimization_parallel, fastmath=True)(self._perf, self._perf_len, self.cl_radii, self.cl_centers, None, self.optimization_tolerance)


        # use only max amount of spheres
        if self.max_spheres_used != -1:
            if self.verbosity > 1:
                print(f'Spheres without capping: {len(self.cl_radii)}')
            
            self.cl_centers, self.cl_radii = self.cl_centers[:self.max_spheres_used], self.cl_radii[:self.max_spheres_used]
            
        if self.verbosity > 1:
            print(f'N-Spheres produced for classification: {len(self.cl_radii)}')

        
        return self
    
    @staticmethod
    @nb.njit(parallel=True, fastmath=True)
    def _predict(X, return_distances, cl_centers, cl_radii, _metric, p, progress):
        """
        Predict the class of the data.
        :param X: The input
        :param return_distances: Return max distance inside and min distance outside (delta from radius).
        :return: If is inside class as bool
        """
        
        # length of X
        length = X.shape[0]

        # create predictions (false by default)
        y = np.full(length, False, dtype=np.bool_)
        
        # if return distances max dist inside and min dist outside
        if return_distances:
            # between [0, +inf[
            # so fill min with large values and max with zeros
            min_dist_outside = np.full(length, 1e8)
            max_dist_inside = np.zeros(length)

        # loop through all rows
        for i in nb.prange(length):
            # update progress
            if progress is not None:
                progress.update(1)
            
            # loop till one classifier is true
            for j in range(cl_centers.shape[0]):
                # WHY THIS MESSY CODE (UGLY WORKAROUND) AND NOT JUST A METRIC FUNCTION POINTER?
                # the numba compiler seems to not be able to optimize
                # good enough when using function pointers
                # At least I had no success with it.
                if _metric == 0:
                    dist = np.std(cl_centers[j] - X[i])  # std
                elif _metric == 1:
                    dist = np.sum((cl_centers[j] - X[i]) ** p) ** (1. / p) # minkowski
                elif _metric == 2:
                    dist = np.sqrt(np.sum((cl_centers[j] - X[i]) ** 2.))  # euclid
                elif _metric == 3:
                    dist = np.sum(np.abs(cl_centers[j] - X[i])) / cl_centers[j].shape[0]  # hamming
                elif _metric == 4:
                    dist = np.max(np.abs(cl_centers[j] - X[i]))  # max
                elif _metric == 5:
                    dist = 1 - np.sum(cl_centers[j] * X[i]) / (np.sqrt(np.sum(cl_centers[j] ** 2)) * np.sqrt(np.sum(X[i] ** 2)) + 1e-8)  # cosine
                elif _metric == 6:
                    dist = 1 - np.sum(np.minimum(cl_centers[j], X[i])) / (np.sum(np.maximum(cl_centers[j], X[i])) + 1e-8)  # jaccard
                elif _metric == 7:
                    dist = 1 - 2 * np.sum(np.minimum(cl_centers[j], X[i])) / (np.sum(cl_centers[j]) + np.sum(X[i]) + 1e-8)  # dice
                elif _metric == 8: 
                    dist = np.sum(np.abs(cl_centers[j] - X[i]) / (np.abs(cl_centers[j]) + np.abs(X[i]) + 1e-8))  # canberra
                elif _metric == 9:
                    dist = np.sum(np.abs(cl_centers[j] - X[i])) / (np.sum(np.abs(cl_centers[j] + X[i])) + 1e-8)  # braycurtis
                elif _metric == 10:
                    dist = 1 - np.corrcoef(cl_centers[j], X[i])[0, 1]  # correlation
                elif _metric == 11:
                    dist = np.sum(cl_centers[j] * X[i]) / (np.sum(np.abs(cl_centers[j] - X[i])) + 1e-8)  # yule
                elif _metric == 12:
                    dist = np.arccos(np.sum(cl_centers[j] * X[i]) / (np.sqrt(np.sum(cl_centers[j] ** 2)) * np.sqrt(np.sum(X[i] ** 2)) + 1e-8)) / np.pi  # havensine
                elif _metric == 13:
                    dist = np.sum(cl_centers[j] - X[i])  # sum
                elif _metric == 14:
                    dist = np.sum((cl_centers[j] - X[i]) ** 2) / X[i].shape[0]  # mse
                elif _metric == 15:
                    dist = 1. / np.sum((cl_centers[j] - X[i]) ** 2. + 1e-8)  # ots
                    
                
                
                if return_distances: 
                    # if return distances loop through whole loop
                    # in order to find max inside distance and min outside distance
                    # get delta from radius
                    dist -= cl_radii[j] 
                    # if is inside radius
                    if dist < 0:
                        y[i] = True
                         
                        # if absolute delta is larger than current max inside, replace
                        dist = abs(dist)
                        if dist > max_dist_inside[i]:
                            max_dist_inside[i] = dist  
                    else:
                        # if delta is smaller than current min outside, replace
                        if dist < min_dist_outside[i]:
                            min_dist_outside[i] = dist
                else:
                    # if not returning distances
                    # don't loop whole loop
                    # if is inside (dist smaller than radius)
                    if dist < cl_radii[j]:
                        y[i] = True
                        break

        return y, min_dist_outside, max_dist_inside
    
    
    @staticmethod
    @nb.njit(fastmath=True)
    def _reduce_size(centers, radii, _metric, p, lose_rest):
        """
        Reduce the number of spheres used by finding points next to each other
        and taking their average radius and centers.
        """ 
        # define new center and radii lists
        length = len(centers) 
        new_centers, new_radii = [], []
        
        # keep track of which indexes have been used to create new spheres and which not
        used_idx = np.full(length, False)

        # calculate distances between all sphere centers with provided distance measurement
        min_idx = np.full(length, -1)

        for i in range(length):
            min_value = 1e8

            for j in range(length):
                if i == j:
                    continue

                v = 1e8
                # WHY THIS MESSY CODE (UGLY WORKAROUND) AND NOT JUST A METRIC FUNCTION POINTER?
                # the numba compiler seems to not be able to optimize
                # good enough when using function pointers
                # At least I had no success with it.
                if _metric == 0:
                    v = np.std(centers[i] - centers[j])  # std
                elif _metric == 1:
                    v = np.sum((centers[i] - centers[j]) ** p) ** (1. / p) # minkowski
                elif _metric == 2:
                    v = np.sqrt(np.sum((centers[i] - centers[j]) ** 2.))  # euclid
                elif _metric == 3:
                    v = np.sum(np.abs(centers[i] - centers[j])) / centers[i].shape[0]  # hamming
                elif _metric == 4:
                    v = np.max(np.abs(centers[i] - centers[j]))  # max
                elif _metric == 5:
                    v = 1 - np.sum(centers[i] * centers[j]) / (np.sqrt(np.sum(centers[i] ** 2)) * np.sqrt(np.sum(centers[j] ** 2)) + 1e-8)  # cosine
                elif _metric == 6:
                    v = 1 - np.sum(np.minimum(centers[i], centers[j])) / (np.sum(np.maximum(centers[i], centers[j])) + 1e-8)  # jaccard
                elif _metric == 7:
                    v = 1 - 2 * np.sum(np.minimum(centers[i], centers[j])) / (np.sum(centers[i]) + np.sum(centers[j]) + 1e-8)  # dice
                elif _metric == 8: 
                    v = np.sum(np.abs(centers[i] - centers[j]) / (np.abs(centers[i]) + np.abs(centers[j]) + 1e-8))  # canberra
                elif _metric == 9:
                    v = np.sum(np.abs(centers[i] - centers[j])) / (np.sum(np.abs(centers[i] + centers[j])) + 1e-8)  # braycurtis
                elif _metric == 10:
                    v = 1 - np.corrcoef(centers[i], centers[j])[0, 1]  # correlation
                elif _metric == 11:
                    v = np.sum(centers[i] * centers[j]) / (np.sum(np.abs(centers[i] - centers[j])) + 1e-8)  # yule
                elif _metric == 12:
                    v = np.arccos(np.sum(centers[i] * centers[j]) / (np.sqrt(np.sum(centers[i] ** 2)) * np.sqrt(np.sum(centers[j] ** 2)) + 1e-8)) / np.pi  # havensine
                elif _metric == 13:
                    v = np.sum(centers[i] - centers[j])  # sum
                elif _metric == 14:
                    v = np.sum((centers[i] - centers[j]) ** 2) / centers[i].shape[0]  # mse
                elif _metric == 15:
                    v = 1. / np.sum((centers[i] - centers[j]) ** 2. + 1e-8)  # ots

                # compare to previous value
                if v < min_value:
                    min_value = v
                    min_idx[i] = j

        # append average center and radius
        # if they both marked themselves as their nearest neighbors
        for i, j in enumerate(min_idx):
            if i == min_idx[j] and i < j:
                # generate new sphere and add to list
                new_centers.append((centers[i] + centers[j]) / 2.)
                new_radii.append((radii[i] + radii[j]) / 2.)
                # mark indexes as used
                used_idx[i] = True
                used_idx[j] = True
        
        # if lose rest is False, add the unused indexes to list
        if not lose_rest:
            # add old unused indexes to new_list
            for i in range(length):
                if not used_idx[i]:
                    new_centers.append(centers[i])
                    new_radii.append(radii[i])

        # return new centers and radii
        return new_centers, new_radii
    
    

    def reduce_size(self, lose_rest=False, repetitions=1):
        """
        Reduce the number of spheres used by finding points next to each other
        and taking their average radius and centers.

        :param repetitions: how often to repeat the algorithm
        :param lose_rest: if should keep only newly generated spheres and lose the for generating unused ones
        :return: self
        """

        for _ in range(repetitions):
            # perform algorithm
            new_centers, new_radii = self._reduce_size(self.cl_centers, self.cl_radii, self._metric, self.p, lose_rest)

            # convert reflective list to np array
            self.cl_centers = np.array(new_centers)
            self.cl_radii = np.array(new_radii)
        
        
        
        if self.verbosity > 0:
            print(f'New size: {len(self.cl_centers)}')
            
        return self
    
   
    def predict(self, X, return_distances=False):
        """
        Predict the class of the data and scale data before.
        :param X: The input
        :param return_distances: Return max distance inside and min distance outside as well.
        :return: The predicted class
        """
        # transform X as defined
        if self.standard_scaling:
            X = self.standard_scaler.transform(X)
            
        if self.normalization:
            X = self.normalizer.transform(X)
         
        # use _predict numba method
        if self.verbosity > 0:
            with ProgressBar(total=X.shape[0]) as progress:
                res = self._predict(X, return_distances, self.cl_centers, self.cl_radii, self._metric, self.p, progress) 
        else:
            res = self._predict(X, return_distances, self.cl_centers, self.cl_radii, self._metric, self.p, None)
        
        
        # if only return y, do so
        return res if return_distances else res[0]
    
    def score(self, X, y):
        """
        Score the model
        :param X: The input data
        :param y: The target data
        :return: The accuracy
        """

        assert X.shape[0] == y.shape[0]

        return (self.predict(X) == (y == self.in_class)).sum() / y.shape[0]


    def dump(self, file):
        """
        dump classifier as file
        :param file: file to dump to
        """
        with open(file, "wb", protocol=4) as f:
            pickle.dump(self, f)


    def load(self, file):
        """
        load classifier from file
        :param file: file to load from
        """
        with open(file, "rb") as f:
            self = pickle.load(f)

        return self

    def __repr__(self):
        return f"SphereNet(metric={self.metric}, in_class={self.in_class}, standard_scaling={self.standard_scaling}, normalization={self.normalization})"

    def __str__(self):
        return self.__repr__()




class MultiSphereNet:
    """
    MultiSphereNet for multi class classification
    Init with a list of SphereNet classifiers and shares normalizers and standard scalers
    """
    
    def __init__(self, standard_scaling=False, normalization=False, remove_training_outliers=True, pred_mode='conservative', verbosity=0, **kwargs):
        """
        MultiSphereNet
        
        Prediction Modes:
        - conservative: Fastest classification mode. Points classified can be overwritten, if one inside is found, point will be marked. -1 value for not classified points.
        - careful: If there are classification conflicts (points classified by multiple classifiers), those will be resolved by comparing max distance to sphere shape. -1 value for not classified points.
        - force: Same as careful mode, but a class is forced by min distance when not classified by any SphereNet.
        
        
        :param standard_scaling: See SphereNet.
        :param normalization: See SphereNet.
        :param remove_training_outliers: See SphereNet.
        :param pred_mode: Prediction mode. Available: ['conservative', 'careful', 'force']
        :param verbosity: logging level
        :param kwargs: see SphereNet
        """

        self.standard_scaling = bool(standard_scaling)
        self.normalization = bool(normalization)
        self.remove_training_outliers = remove_training_outliers
        self.pred_mode = pred_mode
        self._pred_mode = pred_modes.index(pred_mode)
        self.verbosity = verbosity
        self.kwargs = kwargs
        self.sphere_nets = []


        # init standard scaler and normalizer if needed
        if standard_scaling:
            if isinstance(standard_scaling, StandardScaler):
                self.standard_scaler = standard_scaling
            else:
                self.standard_scaler = StandardScaler()
        else:
            self.standard_scaler = None

        if normalization:
            if isinstance(normalization, Normalizer):
                self.normalizer = normalization
            else:
                self.normalizer = Normalizer()
        else:
            self.normalizer = None


    def fit(self, X, y):
        """
        Fit the model
        :param X: The input data
        :param y: The target data
        :return: self
        """

        assert X.shape[0] == y.shape[0]

        # fit standard scaler if needed and transform X
        if self.standard_scaling:
            if not hasattr(self.standard_scaler, 'n_features_in_'):
                self.standard_scaler.fit(X)

            X = self.standard_scaler.transform(X)

        # transform via normalizer if needed
        if self.normalization:
            X = self.normalizer.transform(X)

        # get unique classes
        self.classes = np.unique(y)
        
        # remove outliers if enabled
        # from every single distribution
        if self.remove_training_outliers:
            if self.verbosity > 1:
                print('removing outliers from training data')
            
            # array marking if is no outlier
            not_outliers = np.full(len(X), False)
            
            # for every distribution let LocalOutlierFactor run
            for cl in self.classes:
                cl_idx = y == cl
                not_outliers[cl_idx] = LocalOutlierFactor().fit_predict(X[cl_idx]) == 1
            
            # overwrite X
            X = X[not_outliers]
            y = y[not_outliers]
        
        # fit one SphereNet for each class
        for cl in self.classes:
            net = SphereNet(in_class=cl, standard_scaling=False, normalization=False, remove_training_outliers=False, verbosity=self.verbosity, **self.kwargs)
            net.fit(X, y)
            self.sphere_nets.append(net)

        return self

    def predict(self, X):
        """
        Predict the class of the data and scale data before.
        :param X: The input
        :return: The predicted class (array), -1 if no class was found
        """
        
        # transform X as defined
        if self.standard_scaling:
            X = self.standard_scaler.transform(X)
            
        if self.normalization:
            X = self.normalizer.transform(X)
            
        # length of X
        length = X.shape[0]
         
        # create a array filled with -1
        if self._pred_mode == 0:  # conservative  
            y = np.full(length, -1, dtype=np.int8)
            
            # predict for each SphereNet
            for net in self.sphere_nets:
                y[net.predict(X)] = net.in_class  # set class if SphereNet predicts True for the data point

            return y
        else:  # else careful or force
            # arrays of all classified max_insides and min_outside distances
            insides = np.full((len(self.sphere_nets), length), False)
            max_insides = np.zeros((len(self.sphere_nets), length))
            min_outsides = np.zeros((len(self.sphere_nets), length))
            
            # for later replacing indices, create dict with indices and real class numbers
            # contain -1 = -1 as class for not classified classes in careful mode
            idx_class = {
                -1: -1
            }
            
            # make predictions
            for i, net in enumerate(self.sphere_nets):
                insides[i], min_outsides[i], max_insides[i] = net.predict(X, return_distances=True)
                idx_class[i] = net.in_class
            
            # get inside max elements (argmax)
            # use this as classification
            classification = max_insides.argmax(axis=0)

            # get a not mask of classified values
            not_classified_mask = ~insides.any(axis=0)
            
            # if mode is force class
            # which means not classified points (classification == 0) 
            # should be forced the nearest class upon
            # for this use min_isides
            if self._pred_mode == 2:
                # and replace these indices with the argmin of min_outsides
                classification[not_classified_mask] = min_outsides[:, not_classified_mask].argmin(axis=0)
            elif self._pred_mode == 1:
                # keep not classified values as -1
                classification[not_classified_mask] = -1
            
            # replace indexes with classes
            # from the dict
            # https://stackoverflow.com/questions/33529593/how-to-use-a-dictionary-to-translate-replace-elements-of-an-array
            idx_class_keys, idx_class_values = list(idx_class.keys()), list(idx_class.values()) 
            sort_idx = np.argsort(idx_class_keys)
            idx = np.searchsorted(idx_class_keys, classification, sorter=sort_idx)
            return np.asarray(idx_class_values)[sort_idx][idx]
        
        
    def reduce_size(self, lose_rest=False, repetitions=1):
        """
        Reduce number of spheres in nets, look at single SphereNet.
        
        :param repetitions: repetitions of algorithm
        :param lose_rest: see SphereNet
        :return: self
        """
        
        for net in self.sphere_nets:
            net.reduce_size(lose_rest=lose_rest, repetitions=repetitions)
        
        return self
            

    def score(self, X, y):
        """
        Score the model
        
        :param X: The input data
        :param y: The target data
        :return: The accuracy
        """

        assert X.shape[0] == y.shape[0]

        return (self.predict(X) == y).sum() / y.shape[0]
    
    def dump(self, file):
        """
        dump classifier as file
        
        :param file: file to dump to
        """
        with open(file, "wb", protocol=4) as f:
            pickle.dump(self, f)

    def load(self, file):
        """
        load classifier from file
        
        :param file: file to load from
        """
        with open(file, "rb") as f:
            self = pickle.load(f)

        return self
    
     def describe(self):
        """
        :return: self description string
        """
        
        out = self.__repr__() + "\n"
        
        # first classifier to read arguments
        fc = self.sphere_nets[0]
        
        for clf in self.sphere_nets:
            out += f"\nNumber of spheres for class {clf.in_class} KeyNet: {len(clf.cl_radii)}"
        
        out += "\n"
        out += dedent(f"""metric={fc.metric} ({fc._metric})
        pred_mode={self.pred_mode} ({self._pred_mode})
        min_dist_scaler={fc.min_dist_scaler}
        min_radius_threshold={fc.min_radius_threshold}
        optimization_tolerance={fc.optimization_tolerance}
        optimization_repetitions={fc.optimization_repetitions}
        optimization_parallel={fc.optimization_parallel}
        min_num_classified={fc.min_num_classified}
        max_spheres_used={fc.max_spheres_used}
        p={fc.p}
        standard_scaling={self.standard_scaling}
        normalization={self.normalization}
        remove_training_outliers={self.remove_training_outliers}
        verbosity={fc.verbosity}""")
        
        return out
    
    def __repr__(self):
        return f"MultiSphereNet with {len(self.sphere_nets)} SphereNets"

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return len(self.sphere_nets)

    def __getitem__(self, item):
        return self.sphere_nets[item]

    def __iter__(self):
        return iter(self.sphere_nets)
