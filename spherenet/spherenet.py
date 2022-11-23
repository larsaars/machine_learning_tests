#!/usr/bin/env python3

import numpy as np
from numba_progress import ProgressBar
import numba as nb
from sklearn.preprocessing import StandardScaler, Normalizer
import pickle


class SphereNet:
    def __init__(self, in_class=1, min_dist_scaler=1.0, min_radius_threshold=0.01, optimization_tolerance=0, max_spheres_used=-1, metric='minkowski', p=2, optimize=True, standard_scaling=False, normalization=False, verbosity=0):
        """
        Initialize the SphereNet model
        
        :param in_class: the class that will be classified as True or inside the spheres
        :param min_dist_scaler: The minimum distance between spheres
        :param min_radius_threshold: The minimum radius of a sphere (if is -1, there is none)
        :param optimization_tolerance: The optimization tolerance (higher = more optimization but less performance)
        :param max_spheres_used: The maximum number of spheres used for classification (-1 = no limit)
        :param metric: The distance metric to be used. Available: ['minkowski', 'hamming', 'max', 'min']
        :param p: the p hyperparam (order of magnitude) when using the minkowski distance (p=1 is manhattan, p=2 is euclid)
        :param optimize: if optimization should be activated
        :param standard_scaling: if should use standard scaling
        :param normalization: if should use normalization
        :param verbosity: The verbosity level (between 0 and 2)
        """
        
        # assign args to self
        self.in_class = in_class
        self.min_dist_scaler = min_dist_scaler
        self.min_radius_threshold = min_radius_threshold
        self.optimization_tolerance = optimization_tolerance
        self.max_spheres_used = max_spheres_used
        self.verbosity = verbosity
        self.optimize = optimize
        self.standard_scaling = standard_scaling
        self.normalization = normalization
        self.metric = metric
        self.p = p
        
        # assign metric function
        if metric == 'minkowski':
            self._metric = self._metric_minkowski
        elif metric == 'euclid':
            self._metric = self._metric_minkowski
            self.p = 2
        elif metric == 'manhattan':
            self._metric = self._metric_minkowski
            self.p = 1
        elif metric == 'hamming':
            self._metric = self._metric_hamming
        elif metric == 'max' or metric == 'chebyshev':
            self._metric = self._metric_max
        elif metric == 'min':
            self._metric = self._metric_min
        elif metric == 'cosine':
            self._metric = self._metric_cosine
        elif metric == 'jaccard':
            self._metric = self._metric_jaccard
        elif metric == 'dice':
            self._metric = self._metric_dice
        elif metric == 'canberra':
            self._metric = self._metric_canberra
        elif metric == 'braycurtis':
            self._metric = self._metric_braycurtis
        elif metric == 'correlation':
            self._metric = self._metric_correlation
        elif metric == 'yule':
            self._metric = self._metric_yule
        elif metric == 'havensine':
            self._metric = self._metric_havensine
        elif metric == 'sum':
            self._metric = self._metric_sum
        elif metric == 'mean':
            self._metric = self._metric_mean
        elif metric == 'median':
            self._metric = self._metric_median
        elif metric == 'std':
            self._metric = self._metric_std

 
        # init scaler(s)
        if standard_scaling:
            self.standard_scaler = StandardScaler()
        if normalization:
            self.normalizer = Normalizer()



    @staticmethod
    @nb.njit(fastmath=True)
    def _metric_std(x1, x2, p) -> np.float64:
        return np.std(x1 - x2)

    @staticmethod
    @nb.njit(fastmath=True)
    def _metric_mean(x1, x2, p) -> np.float64:
        return np.mean(x1 - x2)

    @staticmethod
    @nb.njit(fastmath=True)
    def _metric_median(x1, x2, p) -> np.float64:
        return np.median(x1 - x2)

    @staticmethod
    @nb.njit(fastmath=True)
    def _metric_minkowski(x1, x2, p) -> np.float64:
        return np.sum((x1 - x2) ** p) ** (1. / p)
    
    @staticmethod
    @nb.njit(fastmath=True)
    def _metric_hamming(x1, x2, p) -> np.float64:
        return np.sum(np.abs(x1 - x2)) / x1.shape[0]
    
 
    @staticmethod
    @nb.njit(fastmath=True)
    def _metric_min(x1, x2, p) -> np.float64:
        return np.min(np.abs(x1 - x2))
     
    @staticmethod
    @nb.njit(fastmath=True)
    def _metric_max(x1, x2, p) -> np.float64:
        return np.max(np.abs(x1 - x2))

    @staticmethod
    @nb.njit(fastmath=True)
    def _metric_cosine(x1, x2, p) -> np.float64:
        return 1 - np.sum(x1 * x2) / (np.sqrt(np.sum(x1 ** 2)) * np.sqrt(np.sum(x2 ** 2)) + 1e-8)

    @staticmethod
    @nb.njit(fastmath=True)
    def _metric_jaccard(x1, x2, p) -> np.float64:
        return 1 - np.sum(np.minimum(x1, x2)) / (np.sum(np.maximum(x1, x2)) + 1e-8)

    @staticmethod
    @nb.njit(fastmath=True)
    def _metric_dice(x1, x2, p) -> np.float64:
        return 1 - 2 * np.sum(np.minimum(x1, x2)) / (np.sum(x1) + np.sum(x2) + 1e-8)

    @staticmethod
    @nb.njit(fastmath=True)
    def _metric_canberra(x1, x2, p) -> np.float64:
        return np.sum(np.abs(x1 - x2) / (np.abs(x1) + np.abs(x2) + 1e-8))

    @staticmethod
    @nb.njit(fastmath=True)
    def _metric_braycurtis(x1, x2, p) -> np.float64:
        return np.sum(np.abs(x1 - x2)) / (np.sum(np.abs(x1 + x2)) + 1e-8)

    @staticmethod
    @nb.njit(fastmath=True)
    def _metric_correlation(x1, x2, p) -> np.float64:
        return 1 - np.corrcoef(x1, x2)[0, 1]

    @staticmethod
    @nb.njit(fastmath=True)
    def _metric_yule(x1, x2, p) -> np.float64:
        return np.sum(x1 * x2) / (np.sum(np.abs(x1 - x2)) + 1e-8)
    
    @staticmethod
    @nb.njit(fastmath=True)
    def _metric_havensine(x1, x2, p) -> np.float64:
        return np.arccos(np.sum(x1 * x2) / (np.sqrt(np.sum(x1 ** 2)) * np.sqrt(np.sum(x2 ** 2)) + 1e-8)) / np.pi

    @staticmethod
    @nb.njit(fastmath=True)
    def _metric_sum(x1, x2, p) -> np.float64:
        return np.sum(np.abs(x1 - x2))

    
    @staticmethod
    @nb.njit(parallel=True, fastmath=True)
    def _calculate_spheres(X_IN, X_OUT, progress, min_dist_scaler, min_radius_threshold, metric, p):
        """
        determine n spheres center and their single sphere performance
        :param X_IN: the inside points
        :param X_OUT: the outside points
        :param progress: the progress bar
        :return: the performance, radii and centers for point packages
        """
        
        # arrays have quadratic size 
        length_in, length_out = X_IN.shape[0], X_OUT.shape[0]
        # create arrays to be returned
        perf = np.full((length_in, length_in), False) 
        radii = np.zeros(length_in, dtype=np.float64)
        centers = np.zeros((length_in, X_IN.shape[1]), dtype=np.float64)
        
        # rows to be used (not filtered by threshold)
        rows = np.full(length_in, True, dtype=np.bool_)

        for i in nb.prange(length_in):
            # update progress
            if progress is not None:
                progress.update(1)
            
            # calculate the distance to the nearest outer point
            # since is 2d array perform on loop
            dist = np.zeros(length_out, dtype=np.float64)
            for j in range(length_out):
                dist[j] = metric(X_IN[i], X_OUT[j], p)
            # scale and get min from dist array
            radius = min_dist_scaler * np.min(dist)
             
            # only add if radius is larger than threshold
            if min_radius_threshold == -1 or radius > min_radius_threshold:
                radii[i] = radius
                centers[i] = X_IN[i]
                 
                # calculate performance (how many of the IN points are classified as in by just this one N-sphere)
                # distances < radius of this sphere
                # since is 2d array perform on loop, set each index directly
                for j in range(length_in):
                    perf[i, j] = metric(X_IN[i], X_IN[j], p) < radius
            else:
                rows[i] = False
         
        return perf[rows], radii[rows], centers[rows]


    @staticmethod
    @nb.njit(parallel=True)
    def _remove_ambiguity(perf, radii, centers, progress, optimization_tolerance):
        """
        Remove ambiguity / optimize the spheres by removing overlapping sphere results.
        :param perf: the performance of the spheres
        :param radii: the radii of the spheres
        :param centers: the centers of the spheres
        :param progress: the progress bar
        :return: optimized radii and centers that can be used for classification
        """


        # get len of performances (trues on axis 1)
        perf_len = perf.sum(axis=1)

        # sort performances, radii and centers in the same manner
        # as needed by algorithm
        # sort:
        # small to large (stl)
        # and
        # large to small (lts)
        idx_stl = perf_len.argsort()
        idx_lts = idx_stl[::-1]
        perf_len_stl = perf_len[idx_stl]
        perf_lts = perf[idx_lts]
        perf_stl = perf[idx_stl]
        radii_lts = radii[idx_lts]
        centers_lts = centers[idx_lts]

        rows_len, cols_len = perf_stl.shape
        rows = np.full(
            rows_len, False, dtype=np.bool_
        )  # rows is sorted by lts, true means is disabled, false means enabled

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

        # get centers and radii actually useable for classication
        cl_centers = centers_lts[rows]
        cl_radii = radii_lts[rows]
        return cl_centers, cl_radii


    def fit(self, X, y):
        """
        Fit the SphereNet model to the data
        :param X: The input data
        :param y: The target data 
        :return: The fitted model (self)
        """
        # fit scaler and transform X data in same step
        if self.standard_scaling:
            if self.verbosity > 1:
                print('fitting standard scaler')
                
            X = self.standard_scaler.fit_transform(X)
            
        if self.normalization:
            if self.verbosity > 1:
                print('fitting normalizer')
                
            X = self.normalizer.fit_transform(X)
 
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
                perf, radii, centers = self._calculate_spheres(X_IN, X_OUT, progress, self.min_dist_scaler, self.min_radius_threshold, self._metric, self.p)
        else:
            perf, radii, centers = self._calculate_spheres(X_IN, X_OUT, None, self.min_dist_scaler, self.min_radius_threshold, self._metric, self.p)

        if self.verbosity > 1: 
            print("Removing ambiguity...")

        # remove ambiguity / optimize the spheres
        if self.optimize:
            if self.verbosity > 0:
                with ProgressBar(total=perf.shape[0]) as progress:
                    self.cl_centers, self.cl_radii = self._remove_ambiguity(perf, radii, centers, progress, self.optimization_tolerance)
            else:
                self.cl_centers, self.cl_radii = self._remove_ambiguity(perf, radii, centers, None, self.optimization_tolerance)
        else:
            self.cl_centers, self.cl_radii = centers, radii


        # use only max amount of spheres
        if self.max_spheres_used != -1:
            self.cl_centers, self.cl_radii = self.cl_centers[:self.max_spheres_used], self.cl_radii[:self.max_spheres_used]

        
        return self


    @staticmethod
    @nb.njit(parallel=True, fastmath=True)
    def _predict(X, cl_centers, cl_radii, metric, p):
        """
        Predict the class of the data.
        :param X: The input
        :return: If is inside class as bool
        """
 
        y = np.full(X.shape[0], False, dtype=np.bool_)

        for i in nb.prange(X.shape[0]):
            for j in range(cl_centers.shape[0]):
                if metric(cl_centers[j], X[i], p) < cl_radii[j]:
                    y[i] = True
                    break

        return y
    
    def predict(self, X):
        """
        Predict the class of the data and scale data before.
        :param X: The input
        :return: The predicted class
        """
        # transform X as defined
        if self.standard_scaling:
            X = self.standard_scaler.transform(X)
            
        if self.normalization:
            X = self.normalizer.transform(X)
            
        # use _predict numba method
        return self._predict(X, self.cl_centers, self.cl_radii, self._metric, self.p)

    def score(self, X, y):
        """
        Score the model
        :param X: The input data
        :param y: The target data
        :return: The accuracy
        """

        assert X.shape[0] == y.shape[0]

        return self.predict(X).sum() / y.shape[0]


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
