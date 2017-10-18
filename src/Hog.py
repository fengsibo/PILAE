from skimage.feature import hog
hog_arr = [{'orientations': 9, 'block': (8, 8), 'cell': (3, 3)},
        {'orientations': 9, 'block': (8, 8), 'cell': (3, 3)}
           ]

def extract_hog(arr):
    hog_feature = []
    for i in hog_arr:
        h = hog(arr, i["orientations"], i["block"], i["cell"])
        hog_feature.append(h)

extract_hog(hog_arr)