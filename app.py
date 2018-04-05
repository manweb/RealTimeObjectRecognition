from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import cv2
import time
from utilities import ImageProcessing
from utilities import ImageDiagnostics

def DisplayWelcomeMessage():
    print('************************************************************************')
    print('*                                                                      *')
    print('* This app captures images from a webcam, predicts the object in the   *')
    print('* frame and runs a series of diagnostics on the image. The image       *')
    print('* together with the diagnostic values is displayed live on the screen. *')
    print('*                                                                      *')
    print('* Below is a list of commands available while the app is running to    *')
    print('* to change some display settings and quit the app.                    *')
    print('*                                                                      *')
    print('*    q: Quit the program                                               *')
    print('*    d: Toggle diagnostics information                                 *')
    print('*    v: Change channel for crosshair or turn off                       *')
    print('*    p: Toggle object prediction                                       *')
    print('*                                                                      *')
    print('************************************************************************')

def Run():
    img_proc = ImageProcessing()
    img_diag = ImageDiagnostics()

    # load the pre-trained model ResNet50
    # this model recognizes objects in an image
    model = ResNet50(weights='imagenet')

    # create video capture object. This will access the
    # camera and read the image
    cap = cv2.VideoCapture(0)

    perform_predict = True
    display_diagnostics = True
    crosshair = 1
    # contineously capture the frame of the camera
    # process the image and run some diagnostics on it
    while(True):
        time_start = time.time()
    
        # Capture frame-by-frame
        ret, frame = cap.read()
    
        # Get the scaled and cropped image with the right dimensions to feed to the object predictor
        image_model = img_proc.GetImageForModel(frame)

        x = image.img_to_array(image_model)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
    
        # Get the dimensions of the window which defines the image fed to the model predictor
        object_window = img_proc.GetImageModelBorder(frame)
        panel_width, image_panel = img_proc.GetImageForDisplay(frame, display_diagnostics)
    
        # Display the image diagnostics in the panel
        split_x = int(image_panel.shape[1]/4)
        split_y = int(panel_width/5)
        if display_diagnostics:
            panel_start = image_panel.shape[0] - panel_width + 20
            centroids = img_diag.GetCentroid(frame)
            D4sigma = img_diag.GetD4sigma(frame)
            mean_green = img_diag.GetMeanAndSTD(frame, 1)
            saturated_pixels = img_diag.GetNumberOfSaturatedPixels(frame)
            cv2.putText(image_panel, 'Center X: %.1f'%centroids[2,0], (split_x+10, panel_start), 1, 1, (50,50,255), 1)
            cv2.putText(image_panel, 'Center Y: %.1f'%centroids[2,1], (split_x+10, panel_start+split_y), 1, 1, (50,50,255), 1)
            cv2.putText(image_panel, 'D4sigma X: %.1f'%D4sigma[2,0], (split_x+10, panel_start+2*split_y), 1, 1, (50,50,255), 1)
            cv2.putText(image_panel, 'D4sigma Y: %.1f'%D4sigma[2,1], (split_x+10, panel_start+3*split_y), 1, 1, (50,50,255), 1)
            cv2.putText(image_panel, 'Saturated: %i'%saturated_pixels[2], (split_x+10, panel_start+4*split_y), 1, 1, (50,50,255), 1)
        
            cv2.putText(image_panel, 'Center X: %.1f'%centroids[1,0], (2*split_x+10, panel_start), 1, 1, (50,255,50), 1)
            cv2.putText(image_panel, 'Center Y: %.1f'%centroids[1,1], (2*split_x+10, panel_start+split_y), 1, 1, (50,255,50), 1)
            cv2.putText(image_panel, 'D4sigma X: %.1f'%D4sigma[1,0], (2*split_x+10, panel_start+2*split_y), 1, 1, (50,255,50), 1)
            cv2.putText(image_panel, 'D4sigma Y: %.1f'%D4sigma[1,1], (2*split_x+10, panel_start+3*split_y), 1, 1, (50,255,50), 1)
            cv2.putText(image_panel, 'Saturated: %i'%saturated_pixels[1], (2*split_x+10, panel_start+4*split_y), 1, 1, (50,255,50), 1)
        
            cv2.putText(image_panel, 'Center X: %.1f'%centroids[0,0], (3*split_x+10, panel_start), 1, 1, (255,100,100), 1)
            cv2.putText(image_panel, 'Center Y: %.1f'%centroids[0,1], (3*split_x+10, panel_start+split_y), 1, 1, (255,100,100), 1)
            cv2.putText(image_panel, 'D4sigma X: %.1f'%D4sigma[0,0], (3*split_x+10, panel_start+2*split_y), 1, 1, (255,100,100), 1)
            cv2.putText(image_panel, 'D4sigma Y: %.1f'%D4sigma[0,1], (3*split_x+10, panel_start+3*split_y), 1, 1, (255,100,100), 1)
            cv2.putText(image_panel, 'Saturated: %i'%saturated_pixels[0], (3*split_x+10, panel_start+4*split_y), 1, 1, (255,100,100), 1)
    
            cv2.putText(image_panel, 'Mean green: %.1f'%mean_green[0], (10, panel_start+split_y), 1, 1, (200,200,200), 1)
            cv2.putText(image_panel, 'STD green: %.1f'%mean_green[1], (10, panel_start+2*split_y), 1, 1, (200,200,200), 1)
    
        if crosshair == 1:      # display crosshair for red channel
            channel = 2
            label = 'red'
        elif crosshair == 2:    # display crosshair for green channel
            channel = 1
            label = 'green'
        elif crosshair == 3:    # display crosshair for blue channel
            channel = 1
            label = 'blue'

        if not crosshair == 0 and display_diagnostics:
            line_size = 20
            cv2.line(image_panel, (int(centroids[channel,0])-line_size,int(centroids[channel,1])), (int(centroids[channel,0])+line_size,int(centroids[channel,1])), (200,200,200), 2)
            cv2.line(image_panel, (int(centroids[channel,0]),int(centroids[channel,1])-line_size), (int(centroids[channel,0]),int(centroids[channel,1])+line_size), (200,200,200), 2)
            cv2.putText(image_panel, label, (int(centroids[channel,0])+5, int(centroids[channel,1])-5), 1, 1, (200,200,200), 1)

        # let's predict the object in the frame
        if (perform_predict):
            preds = model.predict(x)
            decoded_preds = decode_predictions(preds, top=3)[0]

            most_likely_object = decoded_preds[0][1]
    
            # draw the window in which the object is recognized and display the most likely object
            cv2.rectangle(image_panel, (object_window[0], object_window[1]), (object_window[2], object_window[3]), (0,0,255), 2)
            if display_diagnostics:
                cv2.putText(image_panel, 'Object: %s'%most_likely_object, (10, panel_start+3*split_y), 1, 1, (255,255,255), 1)
            else:
                cv2.putText(image_panel, most_likely_object, (object_window[0]+5, object_window[1]+15), 1, 1, (0,0,255), 1)

        if display_diagnostics:
            t = time.time() - time_start
            cv2.putText(image_panel, 'Refresh rate: %.2f/s'%(1.0/t), (10, panel_start), 1, 1, (255,255,255), 1)
    
        image_scaled = img_proc.GetScaledImage(image_panel)
    
        # Display the image with diagnostics
        cv2.imshow('frame',image_scaled)
        key_pressed = cv2.waitKey(5)
        if key_pressed == 112:
            if perform_predict:
                perform_predict = False
            else:
                perform_predict = True
        if key_pressed == 100:
            if display_diagnostics:
                display_diagnostics = False
            else:
                display_diagnostics = True
        if key_pressed == 118:
            crosshair += 1
            if crosshair > 3:
                crosshair = 0
        if key_pressed == 113:
            break

    # When users quits, release the capture object
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    DisplayWelcomeMessage()

    Run()
