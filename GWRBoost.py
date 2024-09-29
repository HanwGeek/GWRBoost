import numpy as np
from spreg import user_output

from metric import *

from tqdm import tqdm


class GWRBoost:
    def __init__(
        self,
        y,
        X,
        coords,
        bw,
        nlearner=100,
        inflation=1,
        lr=1,
        spherical=False,
        beta=None,
        aic=False,
        early_stop=False,
        wandb_name=None,
        project_name="GWRBoost",
        verbose=False,
    ):
        X, _, _ = user_output.check_constant(X)

        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.coords = np.array(coords).astype(np.float32)
        self.bw = bw

        self.nsample = y.shape[0]
        self.nbeta = X.shape[1]
        self.nlearner = nlearner
        self.inflation = inflation
        self.lr = lr

        self.beta = beta

        self.aic = aic
        self.wandb_name = wandb_name
        self.project_name = project_name
        self.early_stop = early_stop
        self.verbose = verbose

        if spherical:
            self.dist = self._spherical_local_dist()
        else:
            self.dist = self._local_dist()

        self.weight = self._gaussian_kernel_dist()

    # @profile
    def fit(self):
        # * initilize wandb if `wandb_log` is `True`
        if self.wandb_name is not None:
            import wandb

            wandb.init(project=self.project_name, name=self.wandb_name, reinit=True)

        self.log = {"rss": 0, "r2": 0}

        # * initialize a vectorized GWR process
        # X = np.tile(self.X, reps=(self.nsample, 1, 1))
        y = np.tile(self.y, reps=(self.nsample, 1))[:, :, np.newaxis]
        # xT = np.transpose(X * self.weight[:, :, np.newaxis], (0, 2, 1))
        # xtx = np.matmul(xT, X)
        # xtx_inv_xt = np.linalg.solve(xtx, xT)
        xtx_inv_xt = np.zeros((self.nsample, self.nbeta, self.nsample))
        for i in range(self.nsample):
            xT = (self.X * self.weight[i].reshape(-1, 1)).T
            xtx = np.dot(xT, self.X)
            xtx_inv_xt[i] = np.linalg.solve(xtx, xT)

        beta = np.matmul(xtx_inv_xt, y).squeeze(2)
        y_pred = np.sum(self.X * beta, axis=1)

        self.log["rss"] = RSS(self.y, y_pred)
        self.log["r2"] = R2(self.y, y_pred)

        if self.beta is not None:
            rmse = RMSE(self.beta, beta)
            rmse = dict(zip([f"rmse/beta_{i}" for i in range(len(rmse))], rmse))
            self.log = {**self.log, **rmse}

        if self.wandb_name is not None:
            wandb.log(self.log)

        if self.verbose:
            print(f">>> [1 / {self.nlearner}]")

        # * gradient boosting optimization with `nlearner - 1` linear models
        nlearner = self.nlearner - 1
        # for i in tqdm(range(nlearner)):
        for i in range(nlearner):
            resid = (self.y - y_pred) * self.lr

            resid_beta = np.matmul(
                xtx_inv_xt, np.tile(resid, reps=(self.nsample, 1))[:, :, np.newaxis]
            ).squeeze(2)
            resid_pred = np.sum(self.X * resid_beta, axis=1)

            y_pred += resid_pred
            beta += resid_beta

            r2 = R2(self.y, y_pred)

            if self.early_stop == True and r2 < self.log["r2"]:
                break

            self.log["rss"] = RSS(self.y, y_pred)
            self.log["r2"] = r2

            if self.beta is not None:
                rmse = RMSE(self.beta, beta)
                rmse = dict(zip([f"rmse/beta_{i}" for i in range(len(rmse))], rmse))
                self.log = {**self.log, **rmse}

            if self.verbose:
                print(f">>> [{i + 1} / {self.nlearner}]")

            if self.wandb_name is not None:
                wandb.log(self.log)

        self.niter = i + 1
        self.xtx_inv_xt = xtx_inv_xt
        self.param = beta
        self.y_pred = y_pred
        self.resid = self.y - self.y_pred
        if self.aic:
            tr = self._compute_tr()

            self.log["tr"] = tr
            self.log["aic"] = AIC(self.y, y_pred, tr)
            self.log["aicc"] = AICc(self.y, y_pred, tr)
            self.log["adj_r2"] = Adjusted_R2(self.y, y_pred, tr)

    def _compute_tr(self):
        tr = 0
        # for i in tqdm(range(self.nsample)):
        for i in range(self.nsample):
            H = np.matmul(self.X, self.xtx_inv_xt[i])
            I_H = -H
            I_H[np.arange(self.nsample), np.arange(self.nsample)] += 1
            I_H *= self.lr

            S = np.eye(H.shape[0])
            A = I_H
            S += A
            for _ in range(self.niter - 2):
                A = np.dot(A, I_H)
                S += A

            tr += np.dot(H[i, :], S[:, i])

        return tr

    def _gaussian_kernel_dist(self):
        bandwidth = (
            np.partition(self.dist, int(self.bw) - 1)[:, int(self.bw) - 1] * 1.0000001
        )
        bandwidth = bandwidth[:, np.newaxis]
        kernel = (1 - (self.dist / bandwidth) ** 2) ** 2
        kernel[self.dist >= bandwidth] = 0

        return kernel

    def _spherical_local_dist(self):
        dist = np.radians(self.coords[:, np.newaxis] - self.coords[np.newaxis, :])
        dLat, dLon = dist[:, :, 1], dist[:, :, 0]
        coslat = np.cos(np.radians(self.coords[:, 1]))
        a = np.sin(dLat / 2) ** 2 + np.outer(coslat, coslat) * np.sin(dLon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        R = 6371.0
        return R * c

    def _local_dist(self):
        return np.sqrt(
            np.sum(
                (self.coords[:, np.newaxis] - self.coords[np.newaxis, :]) ** 2, axis=-1
            )
        )


class GWRBoostPro:
    def __init__(
        self,
        y,
        X,
        coords,
        bw,
        nlearner=100,
        inflation=1,
        lr=1,
        spherical=False,
        beta=None,
        aic=False,
        early_stop=False,
        wandb_name=None,
        project_name="GWRBoost",
        verbose=False,
    ):
        X, _, _ = user_output.check_constant(X)

        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.coords = np.array(coords).astype(np.float32)
        self.bw = bw

        self.nsample = y.shape[0]
        self.nbeta = X.shape[1]
        self.nlearner = nlearner
        self.inflation = inflation
        self.lr = lr

        self.beta = beta

        self.aic = aic
        self.wandb_name = wandb_name
        self.project_name = project_name
        self.early_stop = early_stop
        self.verbose = verbose

        if spherical:
            self.dist = self._spherical_local_dist()
        else:
            self.dist = self._local_dist()

        self.weight = self._gaussian_kernel_dist()

    # @profile
    def fit(self):
        # * initilize wandb if `wandb_log` is `True`
        if self.wandb_name is not None:
            import wandb

            wandb.init(project=self.project_name, name=self.wandb_name, reinit=True)

        self.log = {"rss": 0, "r2": 0}

        # * initialize a vectorized GWR process
        X = np.tile(self.X, reps=(self.nsample, 1, 1))
        y = np.tile(self.y, reps=(self.nsample, 1))[:, :, np.newaxis]
        xT = np.transpose(X * self.weight[:, :, np.newaxis], (0, 2, 1))
        xtx = np.matmul(xT, X)
        xtx_inv_xt = np.linalg.solve(xtx, xT)

        beta = np.matmul(xtx_inv_xt, y).squeeze(2)
        y_pred = np.sum(self.X * beta, axis=1)

        self.log["rss"] = RSS(self.y, y_pred)
        self.log["r2"] = R2(self.y, y_pred)

        if self.aic:
            hatmat = np.zeros((self.nsample, self.nsample, self.nsample))
            hatmat[:, np.arange(self.nsample), np.arange(self.nsample)] = 1
            H = np.matmul(X, xtx_inv_xt)  # hat matrix for degree of freedom
            tr = np.trace(H[np.arange(self.nsample), np.arange(self.nsample), :])

            self.log["tr"] = tr
            self.log["aic"] = AIC(self.y, y_pred, tr)
            self.log["aicc"] = AICc(self.y, y_pred, tr)
            self.log["adj_r2"] = Adjusted_R2(self.y, y_pred, tr)

            # resid_H = self.lr * (I - H)
            resid_H = -H
            resid_H[:, np.arange(self.nsample), np.arange(self.nsample)] += 1
            resid_H = self.lr * resid_H

        if self.beta is not None:
            rmse = RMSE(self.beta, beta)
            rmse = dict(zip([f"rmse/beta_{i}" for i in range(len(rmse))], rmse))
            self.log = {**self.log, **rmse}

        if self.wandb_name is not None:
            wandb.log(self.log)

        if self.verbose:
            print(f">>> [1 / {self.nlearner}]")

        # * gradient boosting optimization with `nlearner - 1` linear models
        nlearner = self.nlearner - 1
        for i in range(nlearner):
            resid = (self.y - y_pred) * self.lr

            resid_beta = np.matmul(
                xtx_inv_xt, np.tile(resid, reps=(self.nsample, 1))[:, :, np.newaxis]
            ).squeeze(2)
            resid_pred = np.sum(self.X * resid_beta, axis=1)

            y_pred += resid_pred
            beta += resid_beta

            r2 = R2(self.y, y_pred)

            if self.early_stop == True and r2 < self.log["r2"]:
                break

            self.log["rss"] = RSS(self.y, y_pred)
            self.log["r2"] = r2

            if self.aic:
                hatmat += resid_H
                resid_H = np.matmul(resid_H, resid_H)

                tr = np.trace(
                    np.matmul(H, hatmat)[
                        np.arange(self.nsample), np.arange(self.nsample), :
                    ]
                )

                # self.log["tr"] = tr
                self.log["aic"] = AIC(self.y, y_pred, tr)
                self.log["aicc"] = AICc(self.y, y_pred, tr)
                self.log["adj_r2"] = Adjusted_R2(self.y, y_pred, tr)

            if self.beta is not None:
                rmse = RMSE(self.beta, beta)
                rmse = dict(zip([f"rmse/beta_{i}" for i in range(len(rmse))], rmse))
                self.log = {**self.log, **rmse}

            if self.verbose:
                print(f">>> [{i + 1} / {self.nlearner}]")

            if self.wandb_name is not None:
                wandb.log(self.log)

        self.param = beta
        self.y_pred = y_pred

    def _gaussian_kernel_dist(self):
        bandwidth = (
            np.partition(self.dist, int(self.bw) - 1)[:, int(self.bw) - 1] * 1.0000001
        )
        bandwidth = bandwidth[:, np.newaxis]
        kernel = (1 - (self.dist / bandwidth) ** 2) ** 2
        kernel[self.dist >= bandwidth] = 0

        return kernel

    def _spherical_local_dist(self):
        dist = np.radians(self.coords[:, np.newaxis] - self.coords[np.newaxis, :])
        dLat, dLon = dist[:, :, 1], dist[:, :, 0]
        coslat = np.cos(np.radians(self.coords[:, 1]))
        a = np.sin(dLat / 2) ** 2 + np.outer(coslat, coslat) * np.sin(dLon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        R = 6371.0
        return R * c

    def _local_dist(self):
        return np.sqrt(
            np.sum(
                (self.coords[:, np.newaxis] - self.coords[np.newaxis, :]) ** 2, axis=-1
            )
        )
