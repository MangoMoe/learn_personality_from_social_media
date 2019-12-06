import numpy as np
import pylab

# Num people known
num_people_known = np.array([1, 10, 30, 100, 300, 500, 1000,])
num_people_known_reparam = np.array([1, 10, 30, 100, 300, ])
known_score_avg = np.array([1.2059221902859731, 0.967355299086968, 0.8973274146232854, 0.5216136929520052, 0.3611025877536194, 0.4555275625917818, 0.41859056500658687, ])
known_score_avg_reparam = np.array([1.041766249641065, 1.050192405864935, 0.7791941632173858, 0.42395188386798843, 0.25509356589040183, ])
unknown_score_avg = np.array([2.038324813322264, 1.0740345496178796, 0.9925952038043068, 0.9021760884962429, 1.0616193300665233, 1.007818357652892, 1.1306700745019014, ])

pylab.figure()
pylab.plot(num_people_known, known_score_avg, label="Page scores known")
pylab.plot(num_people_known_reparam, known_score_avg_reparam, label="Page scores known with parameter tuning")
pylab.plot(num_people_known, unknown_score_avg, label="Page scores unknown")
pylab.legend()
pylab.title("Effect of varying numbers of known agents on average error")
pylab.xlabel("Number of agents whose OCEAN scores are known")
pylab.ylabel("Average L2 error of predictions after training")

# number of pages available
num_pages = np.array([1, 10, 30, 100, 300, 500, 1000, ])
num_pages_reparam = np.array([1, 10, 30, 100, 200, 1000, ])
known_score_avg_reparam = np.array([1.2443061876666812, 0.7868478593488349, 0.5013368719480444, 0.2610260917781653, 0.22282479005308145, 0.24798348618709104, ])
known_score_avg = np.array([1.2059221902859731, 0.967355299086968, 0.8973274146232854, 0.5216136929520052, 0.3611025877536194, 0.4555275625917818, 0.41859056500658687, ])
# unknown_score_avg = np.array([2.038324813322264, 1.0740345496178796, 0.9925952038043068, 0.9021760884962429, 1.0616193300665233, 1.007818357652892, 1.1306700745019014, ])

pylab.figure()
pylab.plot(num_pages_reparam, known_score_avg_reparam, label="Page scores known with parameter tuning")
pylab.plot(num_pages, known_score_avg, label="Page scores known")
# pylab.plot(num_pages, unknown_score_avg, label="Page scores unknown")
pylab.legend()
pylab.title("Effect of varying numbers of available pages on average error")
pylab.xlabel("Number of pages available to like")
pylab.ylabel("Average L2 error of predictions after training")

# number of iterations
num_iter = np.array([1, 10, 30, 100, 200, 500, 1000, 2000, ])
num_iter_reparam = np.array([1, 10, 30, 100, 200, 500, 1000, 2000, ])
known_score_avg_reparam = np.array([1.2099365235350095, 1.0820861622483586, 0.909537213249414, 0.575018441236332, 0.29684831965346964, 0.20970046199588405, 0.21036909867113374, 0.2315111435171588, ])
known_score_avg = np.array([1.3217291775992732, 1.1559823881265956, 1.5713546356967467, 0.6499643535619931, 0.5886816818519521, 0.6434879150867281, 0.4365604999160152, 0.6463427413377383, ])
# unknown_score_avg = np.array([2.038324813322264, 1.0740345496178796, 0.9925952038043068, 0.9021760884962429, 1.0616193300665233, 1.007818357652892, 1.1306700745019014, ])

pylab.figure()
pylab.plot(num_iter, known_score_avg, label="Page scores known")
pylab.plot(num_iter_reparam, known_score_avg_reparam, label="Page scores known with parameter tuning")
# pylab.plot(num_pages, unknown_score_avg, label="Page scores unknown")
pylab.legend()
pylab.title("Effect of varying iteration length on average error")
pylab.xlabel("Number of iterations")
pylab.ylabel("Average L2 error of predictions after training")

# window size
num_pages = np.array([1, 2, 5, 10, 30, 100, ])
known_score_avg = np.array([0.4922237207017692, 0.5347954374145065, 0.708493791156376, 0.649096010929638, 0.551064347474634, 0.6432066630161801, ])
# unknown_score_avg = np.array([2.038324813322264, 1.0740345496178796, 0.9925952038043068, 0.9021760884962429, 1.0616193300665233, 1.007818357652892, 1.1306700745019014, ])

pylab.figure()
pylab.plot(num_pages, known_score_avg, label="Page scores known")
# pylab.plot(num_pages, unknown_score_avg, label="Page scores unknown")
pylab.legend()
pylab.title("Effect of varying window size on average error")
pylab.xlabel("Window size (number of data vectors included)")
pylab.ylabel("Average L2 error of predictions after training")

pylab.show()