from pymatgen import Composition, Element
from numpy import zeros, mean

# Training file containing band gaps extracted from Materials Project
# created in previous blog post and linked here
trainFile = open("bandgapDFT.csv","r").readlines()

# Input: pymatgen Composition object
# Output: length-100 vector representing any chemical formula

MAX_Z = 100

def naiveVectorize(composition):
       vector = zeros((MAX_Z))
       for element in composition:
               fraction = composition.get_atomic_fraction(element)
               vector[element.Z - 1] = fraction
       return(vector)