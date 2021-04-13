
class Depth:
  def __init__(self):
    print('Parent Depth obj created...')
  def transform(self, read_from_path, write_to_path):
    print('Depth transform() not set!')

class MonoDepth2(Depth):
  def __init__(self):
    print('MonoDepth2 Depth obj created...')
  def transform(self, read_from_path, write_to_path):
    print('MonoDepth2 transform()')

class MonoDepth3(Depth):
  def __init__(self):
    print('MonoDepth3 Depth obj created...')
  def transform(self, read_from_path, write_to_path):
    print('MonoDepth3 transform()')

depth_module = MonoDepth3()

depth_module.transform()