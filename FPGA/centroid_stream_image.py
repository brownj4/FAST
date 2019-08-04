import numpy as np
from scipy import ndimage
from PIL import Image

###########################################################
# Imager Simulation
###########################################################
# Pixels are reordered in groups of 16, as shown on Page 47, Figure 47
# of the Python 1300 Imager Datasheet: http://www.onsemi.com/pub/Collateral/NOIP1SN1300A-D.PDF
order = [0,2,4,6,1,3,5,7] + [x+8 for x in [7,5,3,1,6,4,2,0]]

# read test image from file
im = np.array(Image.open('real_test_image_1024_by_1280.jpg').convert('L'))
# reorder pixels to simulate transfer from camera to FPGA
im = im.reshape((1024,80,16))
reordered_image = np.zeros(im.shape)
for i in range(16):
  reordered_image[:,:,i] = im[:,:,order[i]]
im = reordered_image.reshape((1024,1280))
'''
import matplotlib.pyplot as plt
plt.imshow(im)
plt.show()
'''

###########################################################
# FPGA Implementation
###########################################################
# reverse camera's reordering of the image in FPGA to fix it prior to processing
im = im.reshape((1024,80,16))
reordered_image = np.zeros(im.shape)
for i in range(16):
  reordered_image[:,:,order[i]] = im[:,:,i]
im = reordered_image.reshape((1024,1280))
# due to memory limitations, only store the last 7 rows of the image
# implemented as a circular buffer, although a shifting queue may work
cache = np.zeros((7,1280), dtype=np.uint32)
# due to memory limitations, only store 7x7 around first 1000 bright pixels
# bright areas are cached instead of being immediately centroided due to
# clock rate limitation of FPGA (must be ~300 MHz to keep up with imager),
# so each clock step's operations must be small (e.g. centroiding may be too complex)
# caching bright areas in RAM should be fast and parallel enough and post-process
# centroiding can be broken up into small steps (e.g. one addition at a time)
bright_areas = np.zeros((1000,7,7), dtype=np.uint32)
# count the number of bright areas identified during image streaming
num_bright_areas = 0
# stars are centroided based on the following diagram of the cached pixels:
# -------XXX------------------------
# ------XOOOX-----------------------
# -----XOOOOOX----------------------
# -----XOOOOOX----------------------
# -----XOOOOOX----------------------
# ------XOOOX-----------------------
# -------XXX-P----------------------
# Legend:
# '-'s are unused pixels from the cache
# 'O's are pixels whose weighted sums are used to compute the star centroid
# 'X's are pixels whose average is used to perform background subtraction
# 'P' is the pixel received from the Imager at this time step
# star centroids consist of y and x offsets, y and x weighted pixel sums, and the pixel sum
centroids = np.zeros((1000,5), dtype=np.uint32)
# track how many pixels are brighter than each power of 2 for use in setting exposure and cutoff
# ranges are >1, >2, >4, >8, >16, >32, >64, >128, >256, and >512
# for example, if there are too many pixel values >512, the exposure would be shortened by the ARM
bright_counts = np.zeros(10)
# pixel value cutoff and exposure time may be configured by the ARM before each image is taken
# pixels brighter than the cutoff are star candidates
cutoff = 10
# stream pixels in one at a time to simulate Imager streaming pixels into FPGA
for im_row in range(1024):
  for im_col in range(1280):
    # receive pixel transmitted from Imager at this time step
    # and overwrite old cached pixel value with new pixel value
    cache[im_row%7,im_col] = im[im_row,im_col]
    # count pixels brighter than each power of two
    for power_of_two in range(10):
      # check whether the current pixel is brighter than the given power of two
      if cache[im_row%7,im_col] > 2 ** power_of_two:
        # increment the counter if the current pixel is brighter
        bright_counts[power_of_two] += 1
    # cache must have its left-most 7x7 of pixels filled to be a valid bright area
    if im_row >= 6:
      if im_col >= 6:
        # center pixel must be brighter than cutoff to be a valid bright area
        center_bright = cache[(im_row-3)%7,im_col-3] > cutoff
        # positions of pixels to the left, above, right, and below the center pixel
        left_above_pixels = [((im_row-3)%7,im_col-4),((im_row-4)%7,im_col-3)]
        right_below_pixels = [((im_row-3)%7,im_col-2),((im_row-2)%7,im_col-3)]
        adjacent_pixels = left_above_pixels + right_below_pixels
        # at least one adjacent pixel must also be brighter than the cutoff
        # to prevent salt noise (i.e. a proton event) from causing false positives
        # adjacent_bright = False
        # for (y_adj, x_adj) in adjacent_pixels:
          # if cache[y_adj, x_adj] > cutoff:
            # adjacent_bright = True
        adjacent_bright = np.any([cache[y_adj,x_adj] > cutoff for (y_adj, x_adj) in adjacent_pixels])
        # and center pixel must be at least as bright as left/above pixels
        # and center pixel must be brighter than right/below pixels
        # the asymmetry is to prevent one star from creating multiple bright areas
        # however, one rare case still exists (when a diagonal pixel is the same)
        # brighter_than_adjacent = True
        # for (y_adj, x_adj) in left_above_pixels:
          # if cache[y_adj, x_adj] > cache[(im_row-3)%7,im_col-3]:
            # brighter_than_adjacent = False
        # for (y_adj, x_adj) in right_below_pixels:
          # if cache[y_adj, x_adj] >= cache[(im_row-3)%7,im_col-3]:
            # brighter_than_adjacent = False
        brighter_than_adjacent = np.all([cache[y_adj,x_adj] <= cache[(im_row-3)%7,im_col-3] for (y_adj,x_adj) in left_above_pixels] + \
                                        [cache[y_adj,x_adj] < cache[(im_row-3)%7,im_col-3] for (y_adj,x_adj) in right_below_pixels])
        # if all requirements are satisfied, store the 7x7 bright area for later centroiding
        if center_bright and adjacent_bright and brighter_than_adjacent:
          # with a circular buffer cache, rows must be reordered during storage
          # if a shifting queue cache is used instead, no reordering is necessary
          for bright_area_row in range(7):
            bright_areas[num_bright_areas, bright_area_row,:] = cache[(im_row-6+bright_area_row)%7,im_col-6:im_col+1]
          # x and y offsets of the centroid are just the position of the bright area
          centroids[num_bright_areas,0] = im_row-6;
          centroids[num_bright_areas,1] = im_col-6;
          # increment counter for number of identified bright areas
          num_bright_areas += 1
# once image has finished streaming, stored bright areas can be slowly centroided
# recall that each 7x7 bright area has the following centroid structure:
# --XXX--
# -XOOOX-
# XOOOOOX
# XOOOOOX
# XOOOOOX
# -XOOOX-
# --XXX--
# Legend:
# '-'s are unused pixels
# 'O's are pixels whose weighted sums are used to compute the star centroid
# 'X's are pixels whose average is used to perform background subtraction
# mask for pixels used in background subtraction
background_mask = np.array([[0, 0, 1, 1, 1, 0, 0],
                            [0, 1, 0, 0, 0, 1, 0],
                            [1, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 0, 1],
                            [0, 1, 0, 0, 0, 1, 0],
                            [0, 0, 1, 1, 1, 0, 0]])
# mask for pixels used in centroid calculation
centroid_mask = np.array([[0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 1, 1, 0, 0],
                          [0, 1, 1, 1, 1, 1, 0],
                          [0, 1, 1, 1, 1, 1, 0],
                          [0, 1, 1, 1, 1, 1, 0],
                          [0, 0, 1, 1, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0]])
# weights of pixels for calculation of weighted sum along x dimension
x_weights = np.array([[0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 2, 3, 4, 0, 0],
                      [0, 1, 2, 3, 4, 5, 0],
                      [0, 1, 2, 3, 4, 5, 0],
                      [0, 1, 2, 3, 4, 5, 0],
                      [0, 0, 2, 3, 4, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0]])
# weights of pixels for calculation of weighted sum along y dimension
y_weights = np.array([[0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 1, 1, 0, 0],
                      [0, 2, 2, 2, 2, 2, 0],
                      [0, 3, 3, 3, 3, 3, 0],
                      [0, 4, 4, 4, 4, 4, 0],
                      [0, 0, 5, 5, 5, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0]])
for bright_area_index in range(num_bright_areas):
  # retrieve a stored bright area for centroiding
  bright_area = bright_areas[bright_area_index]
  '''
  # calculate the sum of pixels used for background subtraction
  background_sum = np.sum(background_mask * bright_area)
  # calculate average of background pixels using bit shift (i.e. no division necessary)
  # possible on FPGA because there are 16 background pixels, which is a power of 2
  background_mean = background_sum >> 4
  '''
  # calculate median pixel value of the 7x7 bright area
  median = 512
  for jump in [256,128,64,32,16,8,4,2,1]:
    # count how many pixel values in the 7x7 are less than the current median estimate
    num_pixels_below_median = np.sum([1 for pixel in bright_area.flat if pixel <= median])
    # check whether the median estimate is correct, too high or too low
    # note that 25 is (7*7)/2+1, so it corresponds to the median of the 7x7 area
    if num_pixels_below_median == 25:
      break
    elif num_pixels_below_median > 25:
      median -= jump
    else:
      median += jump
  # median code is substituted for the old background_mean code
  background_mean = median
  # calculate sum of pixels used for centroiding
  centroid_sum = np.sum(centroid_mask * bright_area)
  # calculate weighted pixel sums in x and y dimensions
  x_weighted_sum = np.sum(x_weights * bright_area)
  y_weighted_sum = np.sum(y_weights * bright_area)
  # subtract background mean from centroid sum, avoiding underflow
  # which occurs when background is actually brighter than the bright area
  # note that 21 is the number of pixels used for centroiding
  # so this is equivalent to subtracting the mean from each centroid pixel
  centroid_sum = max(0, centroid_sum - 21 * background_mean)
  # subtract background mean from weighted sums, avoiding underflow
  # which occurs when background is actually brighter than the bright area
  # note that 63 = 3*1 + 5*2 + 5*3 + 5*4 + 3*5, the sum of y_weights/x_weights
  # so this is equivalent to subtracting the mean from each centroid pixel
  y_weighted_sum = max(0, y_weighted_sum - 63 * background_mean)
  x_weighted_sum = max(0, x_weighted_sum - 63 * background_mean)
  # store the calculated centroid values for later processing in the ARM
  centroids[bright_area_index,2] = y_weighted_sum
  centroids[bright_area_index,3] = x_weighted_sum
  centroids[bright_area_index,4] = centroid_sum
print("FPGA sends all of the following data to the ARM:")
print(bright_counts)
print(num_bright_areas)
print(centroids[:num_bright_areas])
print("")

###########################################################
# ARM Implementation
###########################################################
# each centroid consists of the star's brightness and its y and x sub-pixel coordinates
centroid_floats = np.zeros((num_bright_areas,3))
# iterate over centroid values calculated by FPGA and convert them to usable values
for centroid_index in range(num_bright_areas):
  # extract values calculated by FPGA for this centroid
  y_offset, x_offset, y_weighted_sum, x_weighted_sum, centroid_sum = centroids[centroid_index]
  # star brightness has already been calculated by the FPGA, no extra calculation is needed
  centroid_floats[centroid_index,0] = centroid_sum
  # coordinates are calculated as the center of mass of the centroid
  # note that pixel coordinates are considered to be at the upper left of the pixel
  centroid_floats[centroid_index,1] = y_offset + .5 + float(y_weighted_sum) / centroid_sum
  centroid_floats[centroid_index,2] = x_offset + .5 + float(x_weighted_sum) / centroid_sum
# ARM sorts the stars from brightest to dimmest
centroid_floats = centroid_floats[np.argsort(-centroid_floats[:,0]),:]
print("ARM calculates the following values:")
for centroid_index in range(num_bright_areas):
  print("Star of brightness {0:f} at position ({1:.2f},{2:.2f})".format(*centroid_floats[centroid_index]))

