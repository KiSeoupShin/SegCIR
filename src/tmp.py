import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from third_party.open_clip.clip import tokenize

aa = tokenize('a photo of cat with on the table , that is on the floor')

import pdb; pdb.set_trace()