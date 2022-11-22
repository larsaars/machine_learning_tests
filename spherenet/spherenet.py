#!/usr/bin/env python3

import pandas as pd
import numpy as np
import tqdm as tqdm
from numba_progress import ProgressBar
import numba as nb
from sklearn.preprocessing import StandardScaler
import pickle


class SphereNet:
    def __init__(self, min_dist_scaler=1.0, min_radius_threshold=0.01, optimization_tolerance=0, max_spheres_used=-1):
        """
        Initialize the SphereNet model
        :param min_dist_scaler: The minimum distance between spheres
        :param min_radius_threshold: The minimum radius of a sphere
        :param optimization_tolerance: The optimization tolerance (higher = more optimization but less performance)
        :param max_spheres_used: The maximum number of spheres used for classification (-1 = no limit)
        """

        self.min_dist_scaler = min_dist_scaler
        self.min_radius_threshold = min_radius_threshold
        self.optimization_tolerance = optimization_tolerance
        self.max_spheres_used = max_spheres_used

        self.scaler = StandardScaler()


    @staticmethod
    @nb.njit(parallel=True, fastmath=True)
    def _calculate_spheres(X_IN, X_OUT, progress, min_dist_scaler, min_radius_threshold):
        """
        determine n spheres center and their single sphere performance

        :param X_IN: the inside points
        :param X_OUT: the outside points
        :param progress: the progress bar
        :return: the performance, radii and centers for point packages
        """
        
        # arrays have quadratic size 
        length = X_IN.shape[0]
        # create arrays to be returned
        perf = np.full((length, length), False) 
        radii = np.zeros(length, dtype=np.float32)
        centers = np.zeros((length, X_IN.shape[1]))
        
        # rows to be used (not filtered by threshold)
        rows = np.full(length, True, dtype=np.bool_)

        for i in nb.prange(length):
            # update progress
            if progress is not None:
                progress.update(1)
            
            # calculate the distance to the nearest outer point
            radius = min_dist_scaler * np.min(
                np.sqrt(np.sum((X_IN[i] - X_OUT) ** 2., axis=1))
            )

            # only add if radius is larger than threshold
            if radius > min_radius_threshold:
                radii[i] = radius
                centers[i] = X_IN[i]

                # calculate performance (how many of the IN points are classified as in by just this one N-sphere)
                # distances < radius of this sphere
                perf[i] = np.sqrt(np.sum((centers[i] - X_IN) ** 2., axis=1)) < radius 
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


    def fit(self, X, y, in_class=0, verbosity=0, optimize=True):
        """
        Fit the SphereNet model to the data
        :param X: The input data
        :param y: The target data (0 outside, 1 inside)
        :param verbosity: The verbosity level
        :return: The fitted model (self)
        """
        # fit scaler
        self.scaler.fit_transform(X)


        # get in X_IN and X_OUT points
        X_IN = X[y == in_class]
        X_OUT = X[y != in_class]

        if verbosity > 1:
            print("X_IN.shape:", X_IN.shape)
            print("X_OUT.shape:", X_OUT.shape)
            
            print('Calculating radii and centers...')

        # calculate the spheres and their performance
        if verbosity > 0:
            with ProgressBar(total=X_IN.shape[0]) as progress:
                perf, radii, centers = self._calculate_spheres(X_IN, X_OUT, progress, self.min_dist_scaler, self.min_radius_threshold)
        else:
            perf, radii, centers = self._calculate_spheres(X_IN, X_OUT, None, self.min_dist_scaler, self.min_radius_threshold)

        if verbosity > 1: 
            print("Removing ambiguity...")

        # remove ambiguity / optimize the spheres
        if optimize:
            if verbosity > 0:
                with ProgressBar(total=perf.shape[0]) as progress:
                    self.cl_centers, self.cl_radii = self._remove_ambiguity(perf, radii, centers, progress, self.optimization_tolerance)
            else:
                self.cl_centers, self.cl_radii = self._remove_ambiguity(perf, radii, centers, None, self.optimization_tolerance)
        else:
            self.cl_centers, self.cl_radii = centers, radii


        # use only max amount of spheres
        self.cl_centers, self.cl_radii = self.cl_centers[:self.max_spheres_used], self.cl_radii[:self.max_spheres_used]

        
        return self


    @staticmethod
    @nb.njit(parallel=True, fastmath=True)
    def _predict(X, cl_centers, cl_radii):
        """
        Predict the class of the data.

        :param X: The input
        :return: The predicted class
        """


        y = np.full(X.shape[0], False, dtype=np.bool_)

        for i in nb.prange(X.shape[0]):
            for j in range(cl_centers.shape[0]):
                if np.sqrt(np.sum((cl_centers[j] - X[i]) ** 2.)) < cl_radii[j]:
                    y[i] = True
                    break

        return y
    
    def predict(self, X):
        """
        Predict the class of the data and scale data before.
        :param X: The input
        :return: The predicted class
        """
        # transform X
        X = self.scaler.transform(X)
        # use _predict numba method
        return self._predict(X, self.cl_centers, self.cl_radii)

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
