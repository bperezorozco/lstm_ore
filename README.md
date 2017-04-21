LSTMs for ordinal regression

Submission for European Conference in Machine Learning 2017

Run instructions>
1. Unzip dataset folder

exp_code in [ORE07, ORE08_1, ORE08_2, ORE10], corresponding to LSTM, BiDirLSTM, BiDirLSTM with input, MLP
data_id in ['MG', 'heart', 'tide_new']
python ore.py "exp_code" "data_id" False "mode"

2. Train a model
"train" mode: python ore.py "exp_code" "data_id" False "train" n_predictions
Trains a model of type exp_code for model data_id. Weights are stored separately.

"test" mode: python ore.py "exp_code" "data_id" test "mode" n_predictions
REQUIRES: test_files.txt. See example
Loads the trained model given in each line in test_files and runs a forecast with n_predictions and no uncertainty propagation.

"extra" mode: python ore.py "exp_code" "data_id" test "extra" n_predictions
Loads the trained model given in each line in test_files and runs a forecast with n_predictions and uncertainty propagation. Plots attrators and plots with uncertainty bars after using propagationg with uncertainty.

"att_mse" mode: python ore.py "exp_code" "data_id" test "att_mse" n_predictions
REQUIRES: attractors. pkl
