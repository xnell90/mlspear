import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Pre-processing Functions
def min_max(X):
    mins = list(X.min(axis = 0))
    maxs = list(X.max(axis = 0))

    return (mins, maxs)

def mean_std(X):
    means = list(X.mean(axis = 0))
    stds  = list(X.std(axis = 0))

    return(means, stds)

def min_max_scaling(X, mins, maxs):
    for i in range(0, X.shape[1]):
        minimum = mins[i]
        maximum = maxs[i]

        spread = maximum - minimum
        X[:, i] =  (X[:, i] - minimum) / spread

def z_score_scaling(X, means, stds):
    for i in range(0, X.shape[1]):
        mean = means[i]
        std  = stds[i]

        X[:, i] =  (X[:, i] - mean) / std

# Warning: kth must be a valid column, otherwise there will be an
# error.
def ohe(X, kth):
    X = X.astype(object)
    column = X[:, kth].reshape(X.shape[0], 1)
    OHE = __ohe_cv(column)

    if kth == 0:
        return np.hstack((OHE, X[:, 1:]))
    elif kth == X.shape[1] - 1:
        return np.hstack((X[:, 0:kth], OHE))
    else:
        return np.hstack((X[:, 0:kth], OHE, X[:, kth + 1:]))
# $$$$
def __ohe_cv(X):
    entries = [X[i][0] for i in range(0, X.shape[0])]

    value_counts = {}
    for entry in entries:
        if entry in list(value_counts): value_counts[entry] += 1
        else: value_counts[entry] = 1

    num_columns = len(list(value_counts))
    encoding_matrix = np.zeros((X.shape[0], num_columns))

    for i in range(0, len(entries)):
        for j in range(0, num_columns):
            values = list(value_counts)

            if entries[i]  == values[j]: encoding_matrix[i,j] = 1

    return encoding_matrix

def train_validate_test(X):
    split_1 = (X.shape[0] * 7) // 10
    split_2 = split_1 + ((X.shape[0] * 15) // 100)

    X_train = X[0:split_1, :]
    X_valid = X[split_1:split_2, :]
    X_test  = X[split_2:, :]

    return (X_train, X_valid, X_test)

def ReLu(z):
    result = z
    result[z < 0] = 0

    return result

def ReLu_derivative(z):
    result = z
    result[z < 0] = 0
    result[z > 0] = 1

    return result

def PReLu(p, a):
    result = a
    _p_ = p * np.ones(a.shape)
    result[a <= 0] = _p_[a <= 0] * a[a <= 0]

    return result

def PReLu_derivative_a(p, a):
    result = a
    result[a > 0] = 1
    _p_ = p * np.ones(a.shape)
    result[a <= 0] = _p_[a <= 0]

    return result

def PReLu_derivative_p(a):
    result = a
    result[a > 0] = 0

    return result

#  Softmax and Sigmoid combined
def softmax(X):
    if X.shape[1] == 1:
        return 1 / (1 + np.exp(- X))
    else:
        P = np.exp(X)
        col_sum = np.apply_along_axis(np.sum, 1, P)
        col_sum = col_sum.reshape((col_sum.shape[0], 1))
        P = P / col_sum

        return P

# Error Metrics
def accuracy(P, Y):
    diff = np.abs(__round(P) - Y)
    return 1 - np.sum(diff) / (Y.shape[0] * Y.shape[1])

def recall(P, Y):
    return np.sum(__round(P) * Y) / np.sum(Y)

# $$$$
def __round(P):
    if P.shape[1] == 1: return np.round(P)
    else:
        result  = np.zeros(P.shape)
        max_col = P.argmax(axis = 1)

        for row in range(0, result.shape[0]):
            result[row, max_col[row]] = 1

        return result

def sum_squared_error(Y, P):
    return 0.5 * np.sum((Y - P) ** 2)

def cost_entropy(Y, P):
    if P.shape[1] == 1 and Y.shape[1] == 1:
        column = Y * np.log(P) + (1 - Y) * np.log(1 - P)
        result = - np.sum(column)
    else:
        result = - np.sum(Y * np.log(P))

    return result

# ROC plots the ROC curve and prints out the AUC
def roc(P, Y):
    x_fpr = []
    y_tpr = []

    p_thresholds = [1 - i/999 for i in range(0, 1000)]

    for p_threshold in p_thresholds:
        PT = __round_by(p_threshold, P)
        (fpr, tpr) = __tpr_fpr(PT, Y)
        x_fpr.append(fpr)
        y_tpr.append(tpr)

    plt.plot(x_fpr, y_tpr)
    plt.title("ROC Curve From the Model", fontweight = 'bold')
    plt.xlabel("FPR")
    plt.ylabel("TPR")

    auc = 0
    for i in range(0, 999):
        auc += 0.5 * (y_tpr[i + 1] + y_tpr[i]) * (x_fpr[i + 1] - x_fpr[i])

    print("AUC for the ROC Curve: " + str(auc))
    plt.show()

# Hidden functions
def __round_by(p_threshold, P):
    return [int(P[i] >= p_threshold) for i in range(0, len(P))]

def __tpr_fpr(PT, Y):
    (tp, fn) = (0, 0)
    (fp, tn) = (0, 0)

    for i in range(0, len(PT)):
        if PT[i] == Y[i]:
            if PT[i] == 1: tp += 1
            else: tn += 1
        else:
            if PT[i] == 1: fp += 1
            else: fn += 1

    tpr = tp / (tp + fn)
    fpr = fp / (tn + fp)

    return (fpr, tpr)

# ---------------------------------------------------------------------------
# Other Functions

def row_sum(M):
    R = np.sum(M, axis = 0)
    R = R.reshape(1, R.shape[0])

    return R
