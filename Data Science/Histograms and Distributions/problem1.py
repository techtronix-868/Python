import numpy as np
import matplotlib.pyplot as plt


def norm_histogram(hist):
    """
    takes a histogram of counts and creates a histogram of probabilities

    :param hist: list
    :return: list
    """
    list_histograms = len(hist) * [0]

    total = sum(hist)
    for i in range(len(list_histograms)):
        list_histograms[i] = hist[i] / total

    return list_histograms

    pass


def compute_j(histo, width):
    """
    takes histogram of counts, uses norm_histogram to convert to probabilties, it then calculates compute_j for one bin width

    :param histo: list 
    :param width: float
    :return: float
    """
    norm_distribution = norm_histogram(histo)
    total_value = sum(histo)
    square = sum([i ** 2 for i in norm_distribution])
    computed_j = (2.0 - (total_value + 1) * square) / (width * (total_value - 1))
    return computed_j
    pass

    pass


def sweep_n(data, minimum, maximum, min_bins, max_bins):
    """
    find the optimal bin
    calculate compute_j for a full sweep [min_bins to max_bins]
    please make sure max_bins is included in your sweep

    :param data: list
    :param minimum: int
    :param maximum: int
    :param min_bins: int
    :param max_bins: int
    :return: list
    """
    solution_array = []
    for i in range(min_bins, max_bins + 1):
        width = (maximum - minimum) / i
        gram = plt.hist(data, i, (minimum, maximum))[0]
        solution_array.append(compute_j(gram, width))
    return solution_array
    pass


def find_min(l):
    """
    generic function that takes a list of numbers and returns smallest number in that list its index.
    return optimal value and the index of the optimal value as a tuple.

    :param l: list
    :return: tuple
    """
    value = l[0]
    index_optimal = 0
    for j in range(len(l)):
        if l[j] < value:
            value = l[j]
            index_optimal = j

    return value, index_optimal
    pass


if __name__ == '__main__':
    data = np.loadtxt('input.txt')  # reads data from input.txt
    lo = min(data)
    hi = max(data)
    bin_l = 1
    bin_h = 100
    js = sweep_n(data, lo, hi, bin_l, bin_h)
    """
    the values bin_l and bin_h represent the lower and higher bound of the range of bins.
    They will change when we test your code and you should be mindful of that.
    """
    print(find_min(js))
