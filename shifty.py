from PIL import Image, ImageFilter
import numpy as np
from scipy.ndimage import gaussian_filter

# Shifty

# Inputs
# Frames as array of images (all same WxH)
W = 400
H = 300
Frames = [np.array(Image.open('images/hello.png').convert('L')), np.array(Image.open('images/world.png').convert('L'))]
# PixelsShiftPerFrame pixels per shift
PSPF = 10

BG_DIMS = [H, W]
FG_DIMS = [H, W + (len(Frames) - 1) * PSPF]
# Frames[0].filter(ImageFilter.BLUR).show()

def showBW(imArray):
  im = Image.new("L", [W,H])
  im.putdata(np.reshape(np.uint8((imArray)), [H*W, 1]))
  im.show()

randBG = np.random.randint(2, size=BG_DIMS)
randFG = np.random.randint(2, size=FG_DIMS)

randFrames = [ np.maximum(randBG[0:H, 0:W], randFG[0:H, i*PSPF:i*PSPF+W]) for i, f in enumerate(range(1))]
for f in randFrames:
  showBW(f*255)
  showBW(gaussian_filter(f*255, sigma=1))