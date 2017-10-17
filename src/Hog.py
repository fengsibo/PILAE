from skimage.feature import hog
hog_arr = [{'orientations': 9, 'block': (8, 8), 'cell': (3, 3)},
        {'orientations': 9, 'block': (8, 8), 'cell': (3, 3)}
           ]
print(hog_arr[0])
# def extract_hog(arr):