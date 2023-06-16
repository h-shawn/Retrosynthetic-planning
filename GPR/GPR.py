import argparse

import gpflow
from gpflow.mean_functions import Constant
from gpflow.utilities import print_summary
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from utils import transform_data, TaskDataLoader, featurise_mols
from kernels import Tanimoto

path = "./data/"


def main(representation, use_pca, n_trials, use_rmse_conf):
    data_loader = TaskDataLoader(path)
    X_train, X_test, Y_train, Y_test = data_loader.load_property_data()
    print("Data Loaded!")
    if use_pca:
        n_components = 100
    else:
        n_components = None
    m = None

    def objective_closure():
        return -m.log_marginal_likelihood()

    r2_list = []
    rmse_list = []
    mae_list = []

    n_test = len(Y_test)

    rmse_confidence_list = np.zeros((n_trials, n_test))
    mae_confidence_list = np.zeros((n_trials, n_test))

    print('\nBeginning training loop...')

    for i in range(0, n_trials):
        Y_train = Y_train.reshape(-1, 1)
        Y_test = Y_test.reshape(-1, 1)

        _, Y_train, _, Y_test, y_scaler = transform_data(
            X_train, Y_train, X_test, Y_test, n_components=n_components, use_pca=use_pca)

        X_train = X_train.astype(np.float64)
        Y_train = X_train.astype(np.float64)
        X_test = X_test.astype(np.float64)
        Y_test = X_test.astype(np.float64)

        k = Tanimoto()
        m = gpflow.models.GPR(data=(X_train, Y_train), mean_function=Constant(
            np.mean(Y_train)), kernel=k, noise_variance=1)
        # Optimise the kernel variance and noise level by the marginal likelihood
        opt = gpflow.optimizers.Scipy()
        opt.minimize(objective_closure, m.trainable_variables,
                     options=dict(maxiter=10000))
        # mean and variance GP prediction
        y_pred, y_var = m.predict_f(X_test)
        y_pred = y_scaler.inverse_transform(y_pred)
        Y_test = y_scaler.inverse_transform(Y_test)
        # Compute scores for confidence curve plotting.
        ranked_confidence_list = np.argsort(y_var, axis=0).flatten()

        for k in range(len(Y_test)):
            # Construct the RMSE error for each level of confidence
            conf = ranked_confidence_list[0:k+1]
            rmse = np.sqrt(mean_squared_error(Y_test[conf], y_pred[conf]))
            rmse_confidence_list[i, k] = rmse
            # Construct the MAE error for each level of confidence
            mae = mean_absolute_error(Y_test[conf], y_pred[conf])
            mae_confidence_list[i, k] = mae

        # Output Standardised RMSE and RMSE on Train Set
        y_pred_train, _ = m.predict_f(X_train)
        train_rmse_stan = np.sqrt(mean_squared_error(Y_train, y_pred_train))
        train_rmse = np.sqrt(mean_squared_error(y_scaler.inverse_transform(
            Y_train), y_scaler.inverse_transform(y_pred_train)))
        print("\nStandardised Train RMSE: {:.3f}".format(train_rmse_stan))
        print("Train RMSE: {:.3f}".format(train_rmse))

        score = r2_score(Y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(Y_test, y_pred))
        mae = mean_absolute_error(Y_test, y_pred)

        print("\nR^2: {:.3f}".format(score))
        print("RMSE: {:.3f}".format(rmse))
        print("MAE: {:.3f}".format(mae))

        r2_list.append(score)
        rmse_list.append(rmse)
        mae_list.append(mae)

    r2_list = np.array(r2_list)
    rmse_list = np.array(rmse_list)
    mae_list = np.array(mae_list)

    print("\nmean R^2: {:.4f} +- {:.4f}".format(np.mean(r2_list),
          np.std(r2_list)/np.sqrt(len(r2_list))))
    print("mean RMSE: {:.4f} +- {:.4f}".format(np.mean(rmse_list),
          np.std(rmse_list)/np.sqrt(len(rmse_list))))
    print("mean MAE: {:.4f} +- {:.4f}\n".format(np.mean(mae_list),
          np.std(mae_list)/np.sqrt(len(mae_list))))

    # Plot confidence-error curves
    # 1e-14 instead of 0 to stop weirdness with len(Y_test) = 29
    confidence_percentiles = np.arange(1e-14, 100, 100/len(Y_test))

    if use_rmse_conf:

        rmse_mean = np.mean(rmse_confidence_list, axis=0)
        rmse_std = np.std(rmse_confidence_list, axis=0)

        # We flip because we want the most confident predictions on the right-hand side of the plot
        rmse_mean = np.flip(rmse_mean)
        rmse_std = np.flip(rmse_std)
        # One-sigma error bars
        lower = rmse_mean - rmse_std
        upper = rmse_mean + rmse_std

        plt.plot(confidence_percentiles, rmse_mean, label='mean')
        plt.fill_between(confidence_percentiles, lower, upper, alpha=0.2)
        plt.xlabel('Confidence Percentile')
        plt.ylabel('RMSE (nm)')
        plt.ylim([0, np.max(upper) + 1])
        plt.xlim([0, 100*((len(Y_test) - 1) / len(Y_test))])
        plt.yticks(np.arange(0, np.max(upper) + 1, 5.0))
        plt.savefig(
            './{}_confidence_curve_rmse.png'.format(representation))
        plt.show()

    else:
        # We plot the Mean-absolute error confidence-error curves
        mae_mean = np.mean(mae_confidence_list, axis=0)
        mae_std = np.std(mae_confidence_list, axis=0)

        mae_mean = np.flip(mae_mean)
        mae_std = np.flip(mae_std)

        lower = mae_mean - mae_std
        upper = mae_mean + mae_std

        plt.plot(confidence_percentiles, mae_mean, label='mean')
        plt.fill_between(confidence_percentiles, lower, upper, alpha=0.2)
        plt.xlabel('Confidence Percentile')
        plt.ylabel('MAE (nm)')
        plt.ylim([0, np.max(upper) + 1])
        plt.xlim([0, 100 * ((len(Y_test) - 1) / len(Y_test))])
        plt.yticks(np.arange(0, np.max(upper) + 1, 5.0))
        plt.savefig(
            './{}_confidence_curve_mae.png'.format(representation))
        plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--representation', type=str, default='fragprints',
                        help='str specifying the molecular representation. '
                             'One of [fingerprints, fragments, fragprints].')
    parser.add_argument('-pca', '--use_pca', type=bool, default=False,
                        help='If True apply PCA to perform Principal Components Regression.')
    parser.add_argument('-n', '--n_trials', type=int, default=20,
                        help='int specifying number of random train/test splits to use')
    parser.add_argument('-rms', '--use_rmse_conf', type=bool, default=True,
                        help='bool specifying whether to compute the rmse confidence-error curves or the mae '
                             'confidence-error curves. True is the option for rmse.')

    args = parser.parse_args()

    main(args.representation, args.use_pca,
         args.n_trials, args.use_rmse_conf)
