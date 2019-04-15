import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt
from skimage import img_as_float
from skimage.morphology import reconstruction
from skimage.morphology import dilation, disk
from skimage.filters import threshold_triangle
import xlsxwriter
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
from tkinter import Tk
from tkinter.filedialog import askopenfilename
# from pyfnnd import deconvolve

Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
filename = askopenfilename(title='Select a file to process')

# init video capture
cap = cv2.VideoCapture(filename)

# read first frame to get properties
read, image = cap.read()

# convert first frame to gray scale and initialize accumulator for max image
max_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

print('Reading in the video to get location of neurons')
frame_number = 0
# loop through all the frames in the video
while True:
    # read frame and whether or not it read a frame from the video
    ret, frame = cap.read()
    if ret:
        # convert current frame to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # take the maximum picture values from the accumlated max image and the current image
        max_image = np.maximum(frame, max_image)
        frame_number += 1
    else:
        print(f'Captured {frame_number} frames')
        break

print('Max image created, isolating neurons and neuropils')
# convert image values from int8 to float
image = img_as_float(max_image)
# blur the image
image = gaussian_filter(image, 1)

# create seed points in image
seed = np.copy(image)
seed[1:-1, 1:-1] = image.min()
mask = image

# run dilation reconstruction on image. This contains the background.
dilated = reconstruction(seed, mask, method='dilation')

# subtract the dilated background image from the image
subtracted = image - dilated


# init disk kernel for dialtion
selem = disk(3)
# dilate the subtracted image
dilated = dilation(subtracted, selem)

# perform threshold on the image using triangle
thresh = threshold_triangle(dilated)
binary = dilated > thresh

# label discrete objects and assign each labeled area a number.
label_image = label(binary)
# remove small (artifactual) objects from the image
label_image = remove_small_objects(label_image, 300)

# init disk kernel for dilationg to get neuropils
kernel = disk(45)

# perform dilation on the image to get the neuron and the surrounding neuropil
neuropils = dilation(label_image, kernel)
# subtract neuron from dilation to get just the neuron
neuropils = neuropils - label_image

print('neuropils and neurons identified')

print('recapturing video')
# capture the video again
cap = cv2.VideoCapture(filename)

# init dictionary for neurons and the neurons with background subtraction
neurons = {}
neurons_subtracted = {}
frame_number = 0

while True:
    ret, frame = cap.read()
    if ret:
        # convert current frame to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = frame.astype(float)
        # get the properties of the labeled regions
        neuropil_regions = regionprops(neuropils, frame)
        for i, region in enumerate(regionprops(label_image, frame)):
            key = f'Neuron {i}'
            if key not in neurons.keys():
                neurons[key] = np.array([])
                neurons_subtracted[key] = np.array([])
            neurons[key] = np.append(neurons[key], [region.max_intensity])
            neurons_subtracted[key] = np.append(neurons_subtracted[key], [(region.max_intensity - neuropil_regions[i].min_intensity)])
        frame_number += 1
    else:
        print(f'Captured {frame_number} frames')
        break

# init dictionary for dF_f
neurons_df_f = {}
# neurons_deconv = {}
plt.plot(neurons['Neuron 0'])
plt.show()

for i, neuron in enumerate(neurons.keys()):
    baseline = np.mean(neurons[neuron][100:200])
    df_f = (neurons[neuron]-baseline)/baseline
    neurons_df_f[neuron] = df_f
    # n_best, c_best, LL, theta_best = deconvolve(np.array(df_f), dt=0.1, verbosity=0, learn_theta=(0, 1, 1, 1, 0) )
    # neurons_deconv[neuron] = n_best

out_file = f'{filename}-results.xlsx'
print(f'writing results out to {out_file}')

workbook = xlsxwriter.Workbook(out_file)
raw_worksheet = workbook.add_worksheet('Raw')
subtracted_worksheet = workbook.add_worksheet('Subtracted')
df_f_worksheet = workbook.add_worksheet('DF_F')
# deconvolved_worksheet = workbook.add_worksheet('Deconvolved')



raw_worksheet.write_row(0,0,neurons.keys())
subtracted_worksheet.write_row(0,0,neurons.keys())
df_f_worksheet.write_row(0,0,neurons.keys())
# deconvolved_worksheet.write_row(map(lambda num: f'Neuron {num}', neurons.keys()))

for i, neuron in enumerate(neurons.keys()):
    raw_worksheet.write_column(1, i, neurons[neuron])
    subtracted_worksheet.write_column(1, i, neurons_subtracted[neuron])
    df_f_worksheet.write_column(1, i, neurons_df_f[neuron])
    # deconvolved_worksheet.write_column(1, neuron, neurons_deconv[neuron])

workbook.close()
