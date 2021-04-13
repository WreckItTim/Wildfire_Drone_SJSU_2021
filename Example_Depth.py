# @Courtney

import .... # pip install x
import .... # pip install y





class Depth:

  def __init__(self):
    print('Parent Depth obj created...')

  def transform(self, read_from_path, write_to_path):
	print('Depth transform() not set!')





class MonoDepth2(Depth):

  def __init__(self):
    print('MonoDepth2 Depth obj created...')
	self.model = load_model()...

  def transform(self, read_from_path, write_to_path):
    print('MonoDepth2 transform()')
	self.model = self.model.transform()....