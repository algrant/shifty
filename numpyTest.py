import numpy
np = numpy
a = numpy.arange(100, dtype='float64').reshape((10,10))
b = numpy.ones(9).reshape((3,3))*10.5

def addArraysAtPoint(big, small, topLeft):
  tL = topLeft
  # vertical & horizontal bounds
  vB = np.clip([tL[0], tL[0] + small.shape[0]], 0, big.shape[0])
  hB = np.clip([tL[1], tL[1] + small.shape[1]], 0, big.shape[1])
  big[vB[0]:vB[1], hB[0]:hB[1]] += small[vB[0]-tL[0]:vB[1]-tL[0], hB[0]-tL[1]:hB[1]-tL[1]]  

a = numpy.ones(100).reshape((10,10))
numpy.random.shuffle(a)


# print(numpy.ravel_multi_index(np.array([5,5,5]), (10,10,10)))

def randMax(arr):
  indexes = np.where(a.flatten()==np.amax(a.flatten()))
  return np.unravel_index(np.random.choice(indexes[0]), arr.shape)

a = numpy.ones(9).reshape((3,3))*3
b = numpy.arange(9).reshape((3,3))+1.0

i = (slice(0,2),slice(0,2))

print(a[i])
print(b[i])