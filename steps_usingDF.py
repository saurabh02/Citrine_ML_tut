from numpy import zeros, mean
from pymatgen import Composition, Element
# Train linear ridge regression model using naive feature set
from sklearn import linear_model, cross_validation, metrics, ensemble
import pandas as pd
import matplotlib.pyplot as plt

# matplotlib.style.use('ggplot')

# Training file containing band gaps extracted from Materials Project
# created in previous blog post and linked here
df = pd.read_csv('bandgapDFT.csv', header=None, names=['materials', 'bandgaps'])
print df[0:2]
print df.columns

# Input: pymatgen Composition object
# Output: length-100 vector representing any chemical formula

MAX_Z = 100  # maximum length of vector to hold naive feature set


def naiveVectorize(composition):
    vector = zeros((MAX_Z))
    for element in composition:
        fraction = composition.get_atomic_fraction(element)
        vector[element.Z - 1] = fraction
    return (vector)


def extractVectors(x):
    material = Composition(x[0])
    return tuple(naiveVectorize(material))


# Constructing naive feature set and adding it to the DF
df1 = df.copy()
df1['naiveFeatures'] = df1.apply(extractVectors, axis=1)
print df1[0:2]

# Establish baseline accuracy by "guessing the average" of the band gap set
# A good model should never do worse.
baselineError = mean(abs(mean(df1[['bandgaps']]) - df1[['bandgaps']]))
print("The MAE of always guessing the average band gap is: " + str(round(baselineError, 3)) + " eV")

##############################################################################################################

# alpha is a tuning parameter affecting how regression deals with collinear inputs
linear = linear_model.Ridge(alpha=0.5)

cv = cross_validation.ShuffleSplit(len(df1), n_iter=10, test_size=0.1, random_state=0)

scores = cross_validation.cross_val_score(linear, list(df1['naiveFeatures']), df1['bandgaps'], cv=cv,
                                          scoring='mean_absolute_error')

print("The MAE of the linear ridge regression band gap model using the naive feature set is: " \
      + str(round(abs(mean(scores)), 3)) + " eV")

##############################################################################################################

# Let's see which features are most important for the linear model

print(
"Below are the fitted linear ridge regression coefficients for each feature (i.e., element) in our naive feature set")

linear.fit(list(df1['naiveFeatures']), df1['bandgaps'])  # fit to the whole data set; we're not doing CV here

print("element: coefficient")

for i in range(MAX_Z):
    element = Element.from_Z(i + 1)
    print(element.symbol + ': ' + str(linear.coef_[i]))


##############################################################################################################
#
# # Create alternative feature set that is more physically-motivated
#
# physicalFeatures = []
#
def extractphysicalFeatures(x):
    #    for material in x[0]:
    theseFeatures = []
    fraction = []
    atomicNo = []
    eneg = []
    group = []

    for element in Composition(x[0]):
        fraction.append(Composition(x[0]).get_atomic_fraction(element))
        atomicNo.append(float(element.Z))
        eneg.append(element.X)
        group.append(float(element.group))

    # We want to sort this feature set
    # according to which element in the binary compound is more abundant
    mustReverse = False

    if fraction[1] > fraction[0]:
        mustReverse = True

    for features in [fraction, atomicNo, eneg, group]:
        if mustReverse:
            features.reverse()
    theseFeatures.append(fraction[0] / fraction[1])
    theseFeatures.append(eneg[0] - eneg[1])
    theseFeatures.append(group[0])
    theseFeatures.append(group[1])
    # physicalFeatures.append(theseFeatures)
    return tuple(theseFeatures)


df1['physicalFeatures'] = df1.apply(extractphysicalFeatures, axis=1)

scores = cross_validation.cross_val_score(linear, list(df1['physicalFeatures']), df1['bandgaps'], cv=cv,
                                          scoring='mean_absolute_error')

print("The MAE of the linear ridge regression band gap model using the physical feature set is: " \
      + str(round(abs(mean(scores)), 3)) + " eV")

##############################################################################################################

# rfr = ensemble.RandomForestRegressor(n_estimators=10) #try 10 trees in the forest
#
# scores = cross_validation.cross_val_score(rfr, naiveFeatures, bandgaps, cv=cv, scoring='mean_absolute_error')
#
# print("The MAE of the nonlinear random forest band gap model using the naive feature set is: "\
# 	+ str(round(abs(mean(scores)), 3)) + " eV")
#
# scores = cross_validation.cross_val_score(rfr, physicalFeatures, bandgaps, cv=cv, scoring='mean_absolute_error')
#
# print("The MAE of the nonlinear random forest band gap model using the physical feature set is: "\
# 	+ str(round(abs(mean(scores)), 3)) + " eV")
