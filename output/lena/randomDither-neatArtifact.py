from PIL import Image, ImageFilter
import numpy as np
from math import floor
def addArraysAtPoint(big, small, topLeft):
  tL = topLeft
  # vertical & horizontal bounds
  vB = np.clip([tL[0], tL[0] + small.shape[0]], 0, big.shape[0])
  hB = np.clip([tL[1], tL[1] + small.shape[1]], 0, big.shape[1])
  big[vB[0]:vB[1], hB[0]:hB[1]] += small[vB[0]-tL[0]:vB[1]-tL[0], hB[0]-tL[1]:hB[1]-tL[1]]  

def genErrorArray(error):
  os = error*1.0/16.0
  ts = error*3.0/16.0
  return np.array([[os, ts, os],[ts, 0, ts], [os, ts, os]])

def applyError(im, error, index):
  addArraysAtPoint(im, genErrorArray(error), [index[0]-1, index[1]-1])

def applyFancyError(im, dith, error, index):
  tL = index
  errorDispersion = 3
  # vertical & horizontal bounds
  vB = np.clip([tL[0], tL[0] + errorDispersion], 0, im.shape[0])
  hB = np.clip([tL[1], tL[1] + errorDispersion], 0, im.shape[1])
  total = np.sum(dith[vB[0]:vB[1], hB[0]:hB[1]])
  if total != 0:
    errorArray = np.ones((errorDispersion, errorDispersion))*error/total
    addArraysAtPoint(im, errorArray, [index[0] - floor(errorDispersion/2), index[1] - floor(errorDispersion/2)])

def showBW(imArray):
  im = Image.new("L", tuple(np.flip(imArray.shape, 0)))
  im.putdata(np.reshape(np.uint8((imArray)), [imArray.size, 1]))
  im.show()

def saveBW(imArray, file):
  im = Image.new("L", tuple(np.flip(imArray.shape, 0)))
  im.putdata(np.reshape(np.uint8((imArray)), [imArray.size, 1]))
  im.save(file)

im = np.array(Image.open('images/lena1.png').convert('L'))/256.0
total = im.size - floor(np.sum(im))
print(total)
dithered = np.ones(im.shape)

randex = np.arange(im.size)
np.random.shuffle(randex)

def to2D(ind):
  return np.unravel_index(ind, im.shape)

def original(randInd):
  return to2D(randex[randInd])

def to1DRand(index):
  return np.ravel_multi_index(index, im.shape)

def randMin(arr):
  a = arr.flatten()
  indexes = np.where(a==np.amin(a))
  return np.unravel_index(np.random.choice(indexes[0]), arr.shape)

for i in range(total):
  maxIndex = randMin(im)
  darkest = im[maxIndex]
  
  # if darkest < 0.5:
  dithered[maxIndex] = 0
  diff = 0 - darkest
  im[maxIndex] = 10000
  applyFancyError(im, dithered, -diff, maxIndex)
  if i%5000 == 0:
    print(darkest, maxIndex)
    print(diff)
    saveBW(dithered*255, "output/lena/%d.png"%(i/5000))



showBW(dithered*255)