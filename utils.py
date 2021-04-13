# @all
from PIL import Image

def hello():
    print('hello there')

img_size = (960, 720) # tello
img_type = '.png' # idk why (code I first used to snap tello drone photos used .png)

def convertPIL(img):
	if img_type == '.png':
		img = img.convert('RGBA').resize(img_size)
	return img

class Timer:
  def __init__(self):
    self.start = time.time()
    self.last = self.start
  def lap(self):
    self.next = time.time()
    self.delta = self.next - self.last
    self.last = self.next
    return self.delta
  def stop(self, reset=True):
    self.next = time.time()
    self.delta = self.next - self.start
    if reset:
      self.start = self.next
    return self.delta