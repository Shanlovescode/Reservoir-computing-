import os
import time

import numpy as np
from tqdm import tqdm
from src.classes import Reservoir, ReservoirLocal
from itertools import product
from src.helpers import average_valid_time
from multiprocessing import Pool
def test_fun(args):
    types = [type(arg) for arg in args]
    return type
def multiprocessing_test_fun(args):
    pool = Pool()
    iters = 5
    types = list(pool.imap_unordered(test_fun, ((args, i) for i in range(iters))))
    print(types)
    return

def gen_model(args):
    (numerical_model, model_class, reservoir_size, num_regions, local_overlap, random_bias, input_pass, squarenodes,
     spectral_radius, input_weight, bias_weight, leakage, train_noise) = args
    if 'Local' in model_class.__name__:
        reservoir = ReservoirLocal(reservoir_size=reservoir_size, input_size=numerical_model.D,
                              random_bias=random_bias, input_pass=input_pass, squarenodes=squarenodes,
                              spectral_radius=spectral_radius, input_weight=input_weight,
                              bias_weight=bias_weight, leakage=leakage, train_noise=train_noise,
                              num_regions = num_regions, local_overlap = local_overlap)
    else:
        reservoir = Reservoir(reservoir_size=reservoir_size, input_size=numerical_model.D,
                              random_bias=random_bias, input_pass=input_pass, squarenodes=squarenodes,
                              spectral_radius=spectral_radius, input_weight=input_weight,
                              bias_weight=bias_weight, leakage=leakage, train_noise=train_noise)
    model = model_class(numerical_model, reservoir)
    return model

def get_valid_times_and_stability(args):
    (model, train_data, test_data, num_batches, regularizations, train_length, discard_length, sync_length,
     predict_length, num_predictions, cutoff, normalization) = args
    valid_times = np.zeros((regularizations.size, num_predictions))
    stability = np.zeros(valid_times.shape)
    #features = model.get_features(train_data, train_length, discard_length)
    info_mat, target_mat = model.get_train_mats(train_data, train_length, discard_length, num_batches = num_batches)
    for (n, regularization) in enumerate(regularizations):
        model.regularization = regularization
        model.train_from_mats(info_mat, target_mat)
        valid_times[n], stability[n] = \
            average_valid_time(test_data, model, num_predictions=num_predictions,
                               sync_length=sync_length, prediction_length=predict_length,
                               cutoff=0.5, normalization=normalization, check_stability=True)
    return np.stack((valid_times, stability), axis = 0)

def hyperparameter_test(numerical_model, model_class, data_folder = 'data', data_name = 'lorenz_model_2',
                        save_vt = True, test_name = 'test1',
                        reservoir_size = 500, num_regions = 0, local_overlap = 2, random_bias = False,
                        pred_normalization = 1.0,
                        squarenodes = True, input_pass = False, spectral_radii = np.array([0.6]),
                        input_weights = np.array([0.1]), bias_weights = np.array([0.]),
                        leakages = np.array([1.0]), train_noises = np.array([1e-2]),
                        regularizations = np.array([1e-4]), train_length = 100000, discard_length = 1000,
                        sync_length = 100, predict_length = 150, num_trains = 10, num_predictions = 10,
                        num_batches = 100):

    models_shape = (spectral_radii.size, input_weights.size, bias_weights.size,
                       leakages.size, train_noises.size)
    models = np.zeros(models_shape, dtype = object).flatten()
    spectral_radii_grid, input_weights_grid, bias_weights_grid, leakages_grid, train_noises_grid = \
        np.meshgrid(spectral_radii, input_weights, bias_weights, leakages, train_noises)
    print(models.shape)
    #pool = Pool()
    num_models = models.size
    #models[:] = np.array(list(tqdm(pool.imap_unordered(gen_model, ((numerical_model, model_class, reservoir_size,
    #                                    num_regions, local_overlap, random_bias,
    #                                  input_pass, squarenodes, spectral_radius, input_weight, bias_weight,
    #                                  leakage, train_noise) for spectral_radius, input_weight, bias_weight,
    #                                 leakage, train_noise in zip(spectral_radii_grid.flatten(),
    #                                                             input_weights_grid.flatten(),
    #                                                             bias_weights_grid.flatten(),
    #                                                             leakages_grid.flatten(),
    #                                                             train_noises_grid.flatten()))),
    #                               total = num_models, desc = 'Generating models...')))
    models[:] = np.array([gen_model((numerical_model, model_class, reservoir_size,
                                                                    num_regions, local_overlap, random_bias,
                                                                    input_pass, squarenodes, spectral_radius,
                                                                    input_weight, bias_weight,
                                                                    leakage, train_noise)) for
                                                                   spectral_radius, input_weight, bias_weight,
                                                                   leakage, train_noise in
                                                                   tqdm(zip(spectral_radii_grid.flatten(),
                                                                       input_weights_grid.flatten(),
                                                                       bias_weights_grid.flatten(),
                                                                       leakages_grid.flatten(),
                                                                       train_noises_grid.flatten()),
                                                                        total = num_models)])
    #print(spectral_radii_grid.flatten())
    #print([model.reservoir.spectral_radius for model in models])

    #valid_times = np.zeros((spectral_radii.size, input_weights.size, bias_weights.size,
    #                        leakages.size, train_noises.size, regularizations.size, num_trains, num_predictions))
    valid_times = np.zeros((num_trains, num_models, 2, regularizations.size, num_predictions))

    for train_idx in range(num_trains):
        print('Loading data at iteration %d' % train_idx)

        train_filename = '%s_train_%d.csv' % (data_name, train_idx)
        train_data_file = os.path.join(data_folder, train_filename)
        train_data     = np.loadtxt(train_data_file, delimiter = ',')[:train_length + discard_length + 1]

        test_filename = '%s_test_%d.csv' % (data_name, train_idx)
        test_data_file = os.path.join(data_folder, test_filename)
        test_data = np.loadtxt(test_data_file, delimiter=',')

        #with tqdm(total = num_models, desc = 'Testing models for dataset %d' % train_idx) as pbar:
        #    for (i, spectral_radius), (j, input_weight), (k, bias_weight), (l, leakage), (m, train_noise) in \
        #            product(enumerate(spectral_radii), enumerate(input_weights), enumerate(bias_weights),
        #                    enumerate(leakages), enumerate(train_noises)):
        #        features = models[i,j,k,l,m].get_features(train_data, train_length, discard_length)
        #        for (n, regularization) in enumerate(regularizations):
        #            models[i,j,k,l,m].regularization = regularization
        #            models[i,j,k,l,m].train_from_features(train_data, features, train_length, discard_length)
        #            valid_times[i,j,k,l,m,n,train_idx,:] = \
        #                average_valid_time(test_data, models[i,j,k,l,m], num_predictions=num_predictions,
        #                                   sync_length=sync_length, prediction_length=predict_length,
        #                                   cutoff=0.5, normalization=pred_normalization)
        #        pbar.update(1)
        cutoff = 0.5
        output = np.zeros((num_models, 2, regularizations.size, num_predictions))
        #output[:] = np.array(list(tqdm(pool.imap_unordered(get_valid_times_and_stability, ((model, train_data,
        #                                         test_data, num_batches, regularizations, train_length, discard_length,
        #                                         sync_length, predict_length, num_predictions,
        #                                         cutoff, pred_normalization) for model in models)),
        #                                         total = num_models,
        #                                         desc = 'Testing models for dataset %d' % train_idx)))
        output[:] = np.array([get_valid_times_and_stability((model, train_data,
                                                 test_data, num_batches, regularizations, train_length, discard_length,
                                                 sync_length, predict_length, num_predictions,
                                                 cutoff, pred_normalization)) for model in tqdm(models)])
        #output[:] = np.array([get_valid_times((model, train_data,
        #                                                                      test_data, regularizations, train_length,
        #                                                                      discard_length,
        #                                                                      sync_length, predict_length,
        #                                                                      num_predictions,
        #                                                                      cutoff, pred_normalization)) for model in
        #                                                                     models], dtype = object)
        print(output.shape)
        print(valid_times.shape)
        valid_times[train_idx, :] = output


    if save_vt:
        model_name = models[0].__class__.__name__
        file_name = '%s_rho%f_leakage%f_sigma%f.npy' % (model_name, spectral_radii[0],
                                                        leakages[0], input_weights[0])
        save_file_path = os.path.join(data_folder, file_name)
        np.save(save_file_path, valid_times)

        #params_dict = {'spectral_radii': spectral_radii,
        #               'input_weights': input_weights,
        #               'bias_weights': bias_weights,
        #               'leakages': leakages,
        #               'train_noises': train_noises,
        #               'regularizations': regularizations,
        #               'num_trains' : num_trains,
        #               'num_predictions' : num_predictions
        #               }

        #file_name = '%s_%s_params.npy' % (model_name, test_name)
        #save_file_path = os.path.join(data_folder, file_name)
        #np.save(save_file_path, params_dict)

    return valid_times