import numpy as np

feature = np.arange(6, 21)
print(feature)

label = (feature * 3) + 4
print(label)

# modify each value assigned to label by adding a different random floating-point value between -2 and +2.
# Don't rely on broadcasting. Instead, create a noise array having the same dimension as label.

noise = (np.random.random([15]) * 4) - 2
print(noise)
label = label + noise 
print(label)