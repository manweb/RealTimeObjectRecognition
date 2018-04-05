import numpy as np
import cv2

class ImageProcessing(object):
    def __init__(self):
        self.variable = 0

    def GetImageForModel(self, image):
        target_size = 224

        img_w = image.shape[1]
        img_h = image.shape[0]
        img_center_x = int(img_w/2)
        img_center_y = int(img_h/2)
        
        # if the image height or width is odd, add a pixel row when cropping to make it sqaure
        add_x = 0
        add_y = 0
        if not img_w % 2 == 0:
            add_x = 1
        if not img_h % 2 == 0:
            add_y = 1

        # create a sub-image that extents all the way to the shorter dimension
        if img_w > img_h:
            image_cropped = image[:,img_center_x-int(img_h/2):img_center_x+int(img_h/2)+add_y]
            img_dim = img_h
        else:
            image_cropped = image[img_center_y-int(img_w/2):img_center_y+int(img_w/2)+add_x,:]
            img_dim = img_w

        image_cropped_scaled = cv2.resize(image_cropped, (0,0), fx=target_size/float(img_dim), fy=target_size/float(img_dim))

        return image_cropped_scaled
            
# Function which returns the border dimensions of the window defining the section of the image
# used for the object recognition
    def GetImageModelBorder(self, image):
        img_w = image.shape[1]
        img_h = image.shape[0]
        img_center_x = int(img_w/2)
        img_center_y = int(img_h/2)
    
        if img_w > img_h:
            border = [img_center_x-int(img_h/2), 0, img_center_x+int(img_h/2), img_h]
        else:
            border = [0, img_center_y-int(img_w/2), img_w, img_center_y+int(img_w/2)]
        
        return border

    def GetImageForDisplay(self, image, addPanel = True):
        if addPanel:
            panel_width = int(image.shape[0]*0.15)
            image_display_panel = cv2.copyMakeBorder(image, 0, panel_width, 0, 0, cv2.BORDER_CONSTANT, value=(20,20,20))
        else:
            panel_width = 0
            image_display_panel = image

        return panel_width, image_display_panel

    def GetScaledImage(self, image):
        if image.shape[0] > 1000 or image.shape[1] > 1000:
            scaling = 1
            if image.shape[0] > image.shape[1]:
                scaling = 1000.0/image.shape[0]
            else:
                scaling = 1000.0/image.shape[1]

        image_scaled = cv2.resize(image, (0,0), fx=scaling, fy=scaling)

        return image_scaled

class ImageDiagnostics(object):
    def __init__(self):
        self.variable = 0

# Function which returns the mean of all pixel values in the specified
# color channel. The input image is and array of size (n,n,3). The color
# channels are 0 = blue, 1 = green, 2 = red
    def GetMean(self, image, color_channel):
        if color_channel < 0 or color_channel > 2:
            print('Warning: Color channel %i not accepted. Value must be 1,2 or 3'%color_channel)

        return np.mean(image[:,:,color_channel])

# Function which returns the standard deviation of all pixel values
# in the specified color channel. The input image is and array of
# size (n,n,3). The color channels are 0 = blue, 1 = green, 2 = red
    def GetSTD(self, image, color_channel):
        if color_channel < 0 or color_channel > 2:
            print('Warning: Color channel %i not accepted. Value must be 1,2 or 3'%color_channel)

        return np.std(image[:,:,color_channel])

# Function which returns the mean and standard deviation of all pixels
# in the specified color channel. See GetMean() and GetSTD() for details
    def GetMeanAndSTD(self, image, color_channel):
        mean = self.GetMean(image, color_channel)
        std = self.GetSTD(image, color_channel)

        return mean, std
    
# Function which calculates the centroids for x and y direction in all color channels.
# It returns an array of size (m,n) with m = number of channels and n = 2 for the two directions
    def GetCentroid(self, image, color_channel = -1):
        if color_channel == -1:
            centroids = np.zeros((image.shape[2],2))

            for i in np.arange(image.shape[2]):
                centroids[i] = self.GetCentroidForChannel(image[:,:,i])
        else:
            centroids = self.GetCentroidForChannel(image[:,:,color_channel])
    
        return centroids
    
# Function which calculates the centroids for x and y direction for a specific channel
    def GetCentroidForChannel(self, image):
        # collapse matrix in x and y
        image_col_x = np.sum(image, axis=0)
        image_col_y = np.sum(image, axis=1)

        # pixel range in x and y
        pixels_x = np.arange(image_col_x.shape[0])
        pixels_y = np.arange(image_col_y.shape[0])
        
        # calculate centroids in both direction as the weighted average over
        # the pixel values
        centroid_x = np.sum(pixels_x*image_col_x)/float(np.sum(image_col_x))
        centroid_y = np.sum(pixels_y*image_col_y)/float(np.sum(image_col_y))

        return centroid_x, centroid_y

# Function which calculates the D4sigma for x and y direction in all color channels.
# It returns an array of size (m,n) with m = number of channels and n = 2 for the two directions
    def GetD4sigma(self, image, color_channel = -1):
        if color_channel == -1:
            D4sigma = np.zeros((image.shape[2],2))
            
            for i in np.arange(image.shape[2]):
                D4sigma[i] = self.GetD4sigmaForChannel(image[:,:,i])
        else:
            D4sigma = self.GetD4sigmaForChannel(image[:,:,color_channel])
        
        return D4sigma

# Function which calculates the D4sigma for x and y direction for a specific channel
    def GetD4sigmaForChannel(self, image):
        # collapse matrix in x and y
        image_col_x = np.sum(image, axis=0)
        image_col_y = np.sum(image, axis=1)
        
        # pixel range in x and y
        pixels_x = np.arange(image_col_x.shape[0])
        pixels_y = np.arange(image_col_y.shape[0])

        # get the centroids of the image
        centroid_x, centroid_y = self.GetCentroidForChannel(image)

        D4sigma_x = 4*np.sqrt(np.sum(image_col_x*np.square(pixels_x-centroid_x))/float(np.sum(image_col_x)))
        D4sigma_y = 4*np.sqrt(np.sum(image_col_y*np.square(pixels_y-centroid_y))/float(np.sum(image_col_y)))

        return D4sigma_x, D4sigma_y
    
# Function which returns the number of saturated pixels in the given color channel
    def GetNumberOfSaturatedPixels(self, image, color_channel = -1):
        if color_channel == -1:
            sat = np.zeros(image.shape[2])

            for i in np.arange(image.shape[2]):
                sat[i] = self.GetNumberOfSaturatedPixelsForChannel(image[:,:,i])
        else:
            sat = self.GetNumberOfSaturatedPixelsForChannel(image[:,:,color_channel])

        return sat
    
# Function which returns the number of saturated pixels in the selected color channel
    def GetNumberOfSaturatedPixelsForChannel(self, image):
        saturation_value = 255

        return image.size - np.count_nonzero(image-saturation_value)
