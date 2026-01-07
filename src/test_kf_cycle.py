from src.model_functions import *
import numpy as np
D = 3
N = 50
ensemble_members = 10
stride = 3
analysis = np.random.rand(ensemble_members, D + N)
model_forecasts = np.random.rand(ensemble_members, D)
bias_features = np.random.rand(ensemble_members, D // stride, 2*(D+N))
W_M = np.random.rand(2*(D+N), stride)
P_W_M = np.eye(2*(D+N)*stride)

W_M_next, P_W_M_next, ill_conditioned = kf_cycle_from_analysis(analysis, model_forecasts, bias_features, W_M.flatten(), P_W_M, D, stride)