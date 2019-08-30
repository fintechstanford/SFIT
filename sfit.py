import numpy as np
from scipy.stats import norm
from statsmodels.stats.proportion import binom_test


def absolute_loss(predicted_y, true_y):
    """Compute the absolute value of the difference between two given real vectors of size N

    Parameters
    ----------
    predicted_y : numpy array of shape (N, 1)
        The predicted outcomes
    true_y : numpy array of shape (N, )
        The true outcomes

    Returns
    -------
    absolute_loss
        a numpy array of shape (N, )
    """
    return np.abs(true_y - np.squeeze(predicted_y))


def sfit_first_order(model, loss, alpha, beta, x, y, verbose=1):
    """Compute the first-order SFIT method to test what are the first-order significant variables within x toward the
    prediction of y as learned by the model.

    Parameters
    ----------
    model :
        A predictive model that can return predicted y given x by calling its method model.predict
    loss : function
        Function that computes the pointwise loss between 2 numpy arrays of outcomes, its first argument should be the
        predicted outcomes and its second should be the true ones.
    alpha : float
        Significance level of the test
    beta: float
        Regularization amount of the test
    x: numpy array of shape (N, p)
        Input data used to perform the tests
    y: numpy array of shape (N, )
        True outcomes
    verbose: boolean
        The summary of the test procedure is printed if true (default) but no printing if false.
    Returns
    -------
    s_1 : list
        the list of first-order significant variables (indexed from 1 to p)
    c_1 : dictionary
        dictionary whose keys are the first-order significant variables ; for each key, its value is a tuple whose first
        element is the test statistic value and second element is its (1 - alpha)% confidence interval
    u_1 : list
        the list of first-order non-significant variables (indexed from 1 to p)
    p_values : numpy array of shape (p, )
        array containing the p-values associated with each variables.
    second_order_significance : boolean
        If true, indicates the presence of significant second-order effects which suggests to use second-order SFIT.
    """
    c_1 = {}
    s_1 = []
    u_1 = []
    (n, p) = x.shape
    p_values = np.zeros(p-1)
    x_intercept = np.copy(x)
    x_intercept[:, range(1, p)] = 0
    predicted_y = model.predict(x_intercept)
    baseline_model_errors = loss(predicted_y, y)
    for j in range(1, p):
        x_j = np.copy(x)
        indices = [i for i in range(1, p) if i != j]
        x_j[:, indices] = 0
        predicted_y_j = model.predict(x_j)
        model_j_errors = loss(predicted_y_j, y)
        delta_j = (1-beta)*baseline_model_errors - model_j_errors
        n_j = np.sum(delta_j > 0)
        p_values[j-1] = binom_test(n_j, n, 0.5, 'larger')
        if binom_test(n_j, n, 0.5, 'larger') < alpha:
            q = norm.ppf(1-alpha/2, 0, 1)
            lower = int(np.floor((n+1)/2 - q*np.sqrt(n)/2))
            upper = int(np.ceil((n+1)/2 + q*np.sqrt(n)/2))
            ordered_delta_j = np.sort(delta_j)
            s_1.append(j)
            c_1[j] = (np.median(delta_j), (ordered_delta_j[lower], ordered_delta_j[upper]))
        else:
            u_1.append(j)
    if verbose:
        print('Summary of first-order SFIT\n'
              '------------------------------------------------\n'
              'First-order significant variables:')
        for key in c_1.keys():
            print('- Variable {0}:'.format(key))
            print('\t Median: {0}'.format(np.round(c_1[key][0], 2)))
            print('\t {0}% confidence interval: {1}'.format(int(100*(1-alpha)), np.round(c_1[key][1], 2)))
        print('------------------------------------------------\n'
              'First-order non-significant variables: {0}'.format(u_1))

    # Test for presence of any second-order significance:
    second_order_significance = False
    x_first = np.copy(x)
    indices = [i for i in range(1, p) if i in u_1]
    x_first[:, indices] = 0
    predicted_y_first = model.predict(x_first)
    model_first_errors = loss(predicted_y_first, y)
    predicted_y_all = model.predict(x)
    model_all_errors = loss(predicted_y_all, y)
    delta_j = model_first_errors - model_all_errors
    n_j = np.sum(delta_j > 0)
    if binom_test(n_j, n, 0.5, 'larger') < alpha:
        second_order_significance = True
        if verbose:
            print('------------------------------------------------\n'
                  'There are some significant second-order variables: recommended to run second-order SFIT.')
    else:
        if verbose:
            print('------------------------------------------------\n'
                  'There are no significant second-order variables.')
    if verbose:
        print('------------------------------------------------ \n'
              '------------------------------------------------ \n')
    return s_1, c_1, u_1, p_values, second_order_significance


def sfit_second_order(model, loss, alpha, beta, x, y, s_1, u_1, verbose=1):
    """Compute the second-order SFIT method to test what are the second-order significant variables within x toward the
    prediction of y as learned by the model.

    Parameters
    ----------
    model :
        A predictive model that can return predicted y given x by calling its method model.predict
    loss : function
        Function that computes the pointwise loss between 2 numpy arrays of outcomes, its first argument should be the
        predicted outcomes and its second should be the true ones.
    alpha : int
        Significance level of the test
    beta: int
        Regularization amount of the test
    x: numpy array of shape (N, p)
        Input data used to perform the tests
    y: numpy array of shape (N, )
        True outcomes
    s_1 : list
        the list of first-order significant variables as returned by sfit_first_order
    u_1 : list
        the list of first-order non-significant variables as returned by sfit_first_order
    verbose: boolean
        The summary of the test procedure is printed if true (default) but no printing if false.

    Returns
    -------
    s_2: list
        the list of second-order significant variables (indexed from 1 to p)
    c_2 : dictionary
        dictionary whose keys are the second-order significant pairs pf variables ; for each key, its value is a tuple
        whose first element is the test statistic value and second element is its (1 - alpha)% confidence interval
    u_2 : list
        the list of second-order non-significant variables (indexed from 1 to p)
    third_order_significance : boolean
        If true, indicates the presence of significant third-order effects.
    """
    s_2 = []
    c_2 = {}
    u_2 = []
    (n, p) = x.shape
    x_intercept = np.copy(x)
    x_intercept[:, range(1, p)] = 0
    predicted_y_intercept = model.predict(x_intercept)
    baseline_model_errors = loss(predicted_y_intercept, y)
    for j in u_1:
        for k in range(1, p):
            x_jk = np.copy(x)
            indices = [i for i in range(1, p) if (i != j and i != k)]
            x_jk[:, indices] = 0
            predicted_y_jk = model.predict(x_jk)
            model_jk_errors = loss(predicted_y_jk, y)
            if k in s_1:
                x_k = np.copy(x)
                indices = [i for i in range(1, p) if i != k]
                x_k[:, indices] = 0
                predicted_y_k = model.predict(x_k)
                model_k_errors = loss(predicted_y_k, y)
                delta_jk = (1-beta)*model_k_errors - model_jk_errors
            else:
                delta_jk = (1-beta)*baseline_model_errors - model_jk_errors
            n_jk = np.sum(delta_jk > 0)
            if binom_test(n_jk, n, 0.5, 'larger') < alpha:
                q = norm.ppf(1-alpha/2, 0, 1)
                lower = int(np.floor((n+1)/2 - q*np.sqrt(n)/2))
                upper = int(np.ceil((n+1)/2 + q*np.sqrt(n)/2))
                ordered_delta_jk = np.sort(delta_jk)
                if j not in s_2:
                    s_2.append(j)
                c_2[(j, k)] = (np.median(delta_jk), (ordered_delta_jk[lower], ordered_delta_jk[upper]))
        if j not in s_2:
            u_2.append(j)
    if verbose:
        print('Summary of second-order SFIT\n'
              '------------------------------------------------\n'
              'Second-order significant variables:')
        for key in c_2.keys():
            print('- Variable {0}:'.format(key))
            print('\t Median: {0}'.format(np.round(c_2[key][0], 2)))
            print('\t {0}% confidence interval: {1}'.format(int(100*(1-alpha)), np.round(c_2[key][1], 2)))
        print('------------------------------------------------\n'
              'Second-order non-significant variables: {0}'.format(u_2))

    # Test for presence of any third-order significance:
    x_second = np.copy(x)
    indices = [i for i in range(1, p) if i in u_2]
    x_second[:, indices] = 0
    predicted_y_second = model.predict(x_second)
    model_second_errors = loss(predicted_y_second, y)
    predicted_y_all = model.predict(x)
    model_all_errors = loss(predicted_y_all, y)
    delta_j = model_second_errors - model_all_errors
    n_j = np.sum(delta_j > 0)
    third_order_significance = False
    if binom_test(n_j, n, 0.5, 'larger') < 0.05:
        third_order_significance = True
        if verbose:
            print('------------------------------------------------\n'
                  'There are some significant third-order variables: recommended to run third-order SFIT.')
    else:
        if verbose:
            print('------------------------------------------------\n'
                  'There are no significant third-order variables.')
    if verbose:
        print('------------------------------------------------ \n'
              '------------------------------------------------ \n')

    return s_2, c_2, u_2, third_order_significance
