import pylab
import numpy as np

# Naive mean calculation
naive_means = np.array([1.2859089352067556, 1.2881699095093966, 1.272204541966302, 1.232648541196715, 1.2576492645281652, 1.2894596058589147, 1.2614455399271205, 1.3109462003600338, 1.3411797690486411, 1.2806345896734572])

# manually enter the data yo,
methods = ["Baseline", "Naive Clustering", "Mean to Mean", "Least Squares", "Decision Tree", "KNN"]
data = [1.77795162, np.mean(naive_means), 1.18595239, 1.23171697, 1.53574763, 1.6008861]
ypos = np.arange(len(methods))

pylab.figure
pylab.bar(ypos, data, align='center', alpha=0.5, color = ["red", "blue", "green", "orange", "purple", "teal"])
pylab.xticks(ypos, methods, fontsize=20)
pylab.ylabel('MSE\n(averaged over 10 runs)', fontsize=20)
pylab.title('Modeling Methods', fontsize=30)
pylab.show()
