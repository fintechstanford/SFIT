from sklearn.preprocessing import MaxAbsScaler
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import EarlyStopping
from sfit import *
import pickle


def generate_y_given(x, epsilon):
    y = 3 + 4 * x[:, 0] + np.multiply(x[:, 0], x[:, 1]) + 3 * np.multiply(x[:, 2], x[:, 2]) \
        + 2 * np.multiply(x[:, 3], x[:, 4]) + epsilon
    return y


def run_unit_test():
    X_test = np.load('./test/X_test.npy')
    Y_test = np.load('./test/Y_test.npy')
    model = load_model('./test/NN_model.h5')
    with open('./test/sfit_dict_first_order.pkl', 'rb') as fp:
        sfit_dict_first_order = pickle.load(fp)
    with open('./test/sfit_dict_second_order.pkl', 'rb') as fp:
        sfit_dict_second_order = pickle.load(fp)
    alpha = 0.01
    beta = 1e-3
    results_sfit_NN = sfit_first_order(model=model,
                                       loss=absolute_loss,
                                       x=X_test,
                                       y=Y_test,
                                       alpha=alpha,
                                       beta=beta)
    results_sfit_second_order_NN = sfit_second_order(model=model,
                                                     loss=absolute_loss,
                                                     alpha=alpha,
                                                     beta=beta,
                                                     x=X_test,
                                                     y=Y_test,
                                                     s_1=results_sfit_NN[0],
                                                     u_1=results_sfit_NN[2])
    assert results_sfit_NN[1] == sfit_dict_first_order, "Should be equal"
    assert results_sfit_second_order_NN[1] == sfit_dict_second_order, "Should be equal"


if __name__ == "__main__":

    run_unit_test()

    N_train = 100000
    N_test = 20000
    d = 10
    noise = 0.01

    np.random.seed(0)
    tf.compat.v1.random.set_random_seed(0)

    # Generate random train and test set of shape (N_train, d) and (N_test, d):
    X_train = np.random.normal(0, 1, size=(N_train, d))
    epsilon_train = np.random.normal(0, noise, N_train)
    Y_train = generate_y_given(X_train, epsilon_train)
    max_abs_scaler = MaxAbsScaler(copy=False)
    X_train = np.insert(X_train, 0, 1, axis=1)
    X_test = np.random.normal(0, 1, size=(N_test, d))
    epsilon_test = np.random.normal(0, noise, N_test)
    Y_test = generate_y_given(X_test, epsilon_test)
    X_test = np.insert(X_test, 0, 1, axis=1)

    # Fit linear regression:
    model = sm.OLS(Y_train, X_train)
    model_lin = model.fit()
    print('Linear regression test loss: {0} \n'.format(np.round(mean_squared_error(Y_test, model_lin.predict(X_test)),
                                                                2)))

    # Fit 2 hidden layers neural network:
    nhidden1 = 150
    nhidden2 = 50
    batch_size = 32
    nr_epochs = 50
    validation_split = 0.25
    inputs = Input(shape=(d + 1,))
    hidden1 = Dense(nhidden1, activation='relu')(inputs)
    hidden2 = Dense(nhidden2, activation='relu')(hidden1)
    output = Dense(1, activation='linear')(hidden2)
    early_stop = EarlyStopping(monitor='val_loss',
                               min_delta=0.01,
                               patience=5)
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x=X_train,
              y=Y_train,
              batch_size=batch_size,
              epochs=nr_epochs,
              validation_split=validation_split,
              callbacks=[early_stop],
              verbose=1)
    base_loss = mean_squared_error(Y_test, model.predict(X_test))
    print('Neural network test loss: {0} \n'.format(np.round(base_loss, 2)))

    # Run first-order SFIT:
    alpha = 0.01
    beta = 1e-3
    results_sfit_lin = sfit_first_order(model=model_lin,
                                        loss=absolute_loss,
                                        x=X_test,
                                        y=Y_test,
                                        alpha=alpha,
                                        beta=beta)

    results_sfit_NN = sfit_first_order(model=model,
                                       loss=absolute_loss,
                                       alpha=alpha,
                                       beta=None,
                                       x=X_test,
                                       y=Y_test)

    # Run second-order SFIT:
    opt_beta = results_sfit_NN[5]
    results_sfit_second_order_NN = sfit_second_order(model=model,
                                                     loss=absolute_loss,
                                                     alpha=alpha,
                                                     beta=opt_beta,
                                                     x=X_test,
                                                     y=Y_test,
                                                     s_1=results_sfit_NN[0],
                                                     u_1=results_sfit_NN[2])
