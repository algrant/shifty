import numpy as np

def addArraysAtPoint(big, small, topLeft):
  tL = topLeft
  # vertical & horizontal bounds
  vB = np.clip([tL[0], tL[0] + small.shape[0]], 0, big.shape[0])
  hB = np.clip([tL[1], tL[1] + small.shape[1]], 0, big.shape[1])
  big[vB[0]:vB[1], hB[0]:hB[1]] += small[vB[0]-tL[0]:vB[1]-tL[0], hB[0]-tL[1]:hB[1]-tL[1]]  
