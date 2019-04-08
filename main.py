import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import cv2
from scipy.ndimage import gaussian_filter
from skimage import img_as_float
from skimage.morphology import reconstruction
from skimage.morphology import dilation, disk
from skimage.filters import threshold_triangle
import xlsxwriter
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
from skimage.color import label2rgb
import matplotlib.patches as mpatches
from tkinter import Tk
from tkinter.filedialog import askopenfilename, askdirectory
# from pyfnnd import deconvolve

Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
filename = askopenfilename(title='Select a file to process')
dirname = askdirectory(initialdir="~", title='Please select a directory')

cap = cv2.VideoCapture(filename)
read, image = cap.read()

width = image.shape[1]
height = image.shape[0]
channels = image.shape[2]
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

max_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
progress_bar = tqdm(total=length+1)

while True:
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        max_image = np.maximum(frame, max_image)
    else:
        break
    progress_bar.update(1)

image = img_as_float(max_image)
image = gaussian_filter(image, 1)

seed = np.copy(image)
seed[1:-1, 1:-1] = image.min()
mask = image

dilated = reconstruction(seed, mask, method='dilation')

subtracted = image - dilated

selem = disk(3)
dilated = dilation(subtracted, selem)

thresh = threshold_triangle(dilated)
binary = dilated > thresh

label_image = label(binary)
label_image = remove_small_objects(label_image, 300)
image_label_overlay = label2rgb(label_image, image=image)

fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(image, cmap='gray')
for i, region in enumerate(regionprops(label_image)):
    minr, minc, maxr, maxc = region.bbox
    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                              fill=False, edgecolor='red', linewidth=1)

    center_r = minr + (maxr - minr) / 2
    center_c = minc + (maxc - minc) / 2

    ax.add_patch(rect)
    ax.annotate(i, (center_c, center_r), color='red', weight='bold',
                fontsize=6, ha='center', va='center')
kernel = disk(45)

neuropils = dilation(label_image, kernel)
neuropils = neuropils - label_image
plt.imshow(neuropils)
plt.show()

cap = cv2.VideoCapture('gcamp.avi')

neurons = {}
neurons_subtracted = {}

while True:
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        neuropil_regions = regionprops(neuropils, frame)
        for i, region in enumerate(regionprops(label_image, frame)):
            if i not in neurons.keys():
                neurons[i] = np.array([])
                neurons_subtracted[i] = np.array([])
            neurons[i] = np.append(neurons[i], [region.max_intensity])
            neurons_subtracted[i] = np.append(neurons_subtracted[i], [(region.max_intensity - neuropil_regions[i].min_intensity)])
    else:
        break


neurons_df_f = {}
# neurons_deconv = {}

for i, neuron in enumerate(neurons.keys()):
    baseline = np.mean(neurons[neuron][100:200])
    df_f = (neurons[neuron]-baseline)/baseline
    neurons_df_f[neuron] = df_f
    # n_best, c_best, LL, theta_best = deconvolve(np.array(df_f), dt=0.1, verbosity=0, learn_theta=(0, 1, 1, 1, 0) )
    # neurons_deconv[neuron] = n_best

workbook = xlsxwriter.Workbook(f'{filename}-results.xlsx')
raw_worksheet = workbook.add_worksheet('Raw')
subtracted_worksheet = workbook.add_worksheet('Subtracted')
df_f_worksheet = workbook.add_worksheet('DF_F')
# deconvolved_worksheet = workbook.add_worksheet('Deconvolved')


raw_worksheet.write_row(map(lambda num: f'Neuron {num}', neurons.keys()))
subtracted_worksheet.write_row(map(lambda num: f'Neuron {num}', neurons.keys()))
df_f_worksheet.write_row(map(lambda num: f'Neuron {num}', neurons.keys()))
# deconvolved_worksheet.write_row(map(lambda num: f'Neuron {num}', neurons.keys()))

numFrames = len(neurons[0])
for neuron in neurons.keys():
    raw_worksheet.write_column(1, neuron, neurons[neuron])
    subtracted_worksheet.write_column(1, neuron, neurons_subtracted[neuron])
    df_f_worksheet.write_column(1, neuron, neurons_df_f[neuron])
    # deconvolved_worksheet.write_column(1, neuron, neurons_deconv[neuron])

workbook.close()
