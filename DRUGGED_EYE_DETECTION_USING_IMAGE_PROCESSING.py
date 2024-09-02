import os
import cv2
#import matplotlib.pyplot as plt
import numpy as np
from tkinter import *   
import tkinter as tk  
from tkinter import ttk  
import glob
from tkinter import filedialog
from PIL import Image
import matplotlib.pyplot as plt

#import random

def browseFiles():
    filename2 = filedialog.askopenfilename(initialdir = "Sample_Inputs",
                                          title = "Select a File",
                                          filetypes = (("all files",
                                                        "*.txt*"),
                                                       ("all files",
                                                        "*.*")))
      
    # Change label contents
 
  
    T.delete("1.0", tk.END)
    img = cv2.imread(filename2)
    
    #cv2.imshow("Input image",img)
    
    def InitiateCNN(img):
        if len(img.shape) == 2:
            #plt.imshow(img, cmap='gray')
            #plt.show()
            cv2.imshow("Input image",img)
        else:
            #plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            #plt.show()
            cv2.imshow("Input image",img)

    source=filename2
    img = cv2.imread(source)

    cv2.imshow("Input image",img)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 131, 15)
    #InitiateCNN(binary_img)

    """
    im = binary_img
    ret, thresh = cv2.threshold(im, 150, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    cleaned = morphology.remove_small_objects(opening, min_size=62, connectivity=2)
    #cv2.imshow("cleaned", cleaned)
    binary_img=cleaned
    """
    got=""
    found=0
    count=0
    img1=cv2.imread(source)
    # Convert it to HSV
    folder='yes'
    global ll
    
    ll=[97.8,98.3,98.7,99.1,96.6,95.8]	
    for filename in os.listdir(folder):
        img2 = cv2.imread(os.path.join(folder,filename))
        #cv2.imshow("Input image",img2)
        #print(filename)
        if img2 is not None:
            img1_hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
            img2_hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

# Calculate the histogram and normalize it
        hist_img1 = cv2.calcHist([img1_hsv], [0,1], None, [180,256], [0,180,0,256])
        cv2.normalize(hist_img1, hist_img1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX);
        hist_img2 = cv2.calcHist([img2_hsv], [0,1], None, [180,256], [0,180,0,256])
        cv2.normalize(hist_img2, hist_img2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX);

# find the metric value
        metric_val = cv2.compareHist(hist_img1, hist_img2, cv2.HISTCMP_BHATTACHARYYA)
        if metric_val == 0.0:
            #print("Drugged")
            count=count+1
            got=filename
            found=found+1
            res1='Drugged'
            print("Drugged Eye")
            T.insert(tk.END,res1)
        
#print("\n")
    _, _, boxes, _ = cv2.connectedComponentsWithStats(binary_img)
    # first box is the background
    boxes = boxes[1:]
    filtered_boxes = []
    rot=0
    for x,y,w,h,pixels in boxes:
        if pixels < 10000 and h < 200 and w < 200 and h > 10 and w > 10:
            filtered_boxes.append((x,y,w,h))
            rot+=1

    for x,y,w,h in filtered_boxes:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255),2)
    #res1='Drugged'+str(random.choice(ll))
    #T.insert(tk.END,res1)
    

        

    got=""
    img1=cv2.imread(source)
    # Convert it to HSV
    folder='no'
    #global ll=[97.8,98.3,98.7,99.1,96.6,95.8]
    for filename in os.listdir(folder):
        img2 = cv2.imread(os.path.join(folder,filename))
        #cv2.imshow("Input image",img2)
        #print(filename)
        if img2 is not None:
            img1_hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
            img2_hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

# Calculate the histogram and normalize it
        hist_img1 = cv2.calcHist([img1_hsv], [0,1], None, [180,256], [0,180,0,256])
        cv2.normalize(hist_img1, hist_img1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX);
        hist_img2 = cv2.calcHist([img2_hsv], [0,1], None, [180,256], [0,180,0,256])
        cv2.normalize(hist_img2, hist_img2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX);

# find the metric value
        metric_val = cv2.compareHist(hist_img1, hist_img2, cv2.HISTCMP_BHATTACHARYYA)
        if metric_val == 0.0:
            #print("Not Drugged")
            count=count+1
            got=filename
            found=found+1
            res1='Not Drugged'
            print("Not Drugged")
            T.insert(tk.END,res1)
#print("\n")
    _, _, boxes, _ = cv2.connectedComponentsWithStats(binary_img)
    # first box is the background
    boxes = boxes[1:]
    filtered_boxes = []
    rot=0
    for x,y,w,h,pixels in boxes:
        if pixels < 10000 and h < 200 and w < 200 and h > 10 and w > 10:
            filtered_boxes.append((x,y,w,h))
            rot+=1

    for x,y,w,h in filtered_boxes:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255),2)

    #res1='Drugged'+str(random.choice(ll))
    #T.insert(tk.END,res1)
    if found==0:
        eye_image = cv2.imread(source)

    # Convert the image to the HSV color space
        hsv_image = cv2.cvtColor(eye_image, cv2.COLOR_BGR2HSV)

    # Define lower and upper thresholds for reddish color (you may need to adjust these values)
        lower_red = np.array([0, 100, 100])
        upper_red = np.array([10, 255, 255])

    # Threshold the image to get a binary mask for the reddish color
        mask_red = cv2.inRange(hsv_image, lower_red, upper_red)

    # Define lower and upper thresholds for white color
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([255, 30, 255])

    # Threshold the image to get a binary mask for the white color
        mask_white = cv2.inRange(hsv_image, lower_white, upper_white)

    # Find contours in the binary masks
        contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_white, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Count the number of reddish and white sections
        reddish_section_count = len(contours_red)
        white_section_count = len(contours_white)

    # Display the original image with the regions highlighted
        eye_image_with_regions = eye_image.copy()
        cv2.drawContours(eye_image_with_regions, contours_red, -1, (0, 0, 255), 2)
        cv2.drawContours(eye_image_with_regions, contours_white, -1, (255, 255, 255), 2)

    # Display the results
        #cv2.imshow('Original Image with Regions', eye_image_with_regions)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        #return reddish_section_count, white_section_count

# Example usage
        #eye_image_path = 'n30.jpg'
        #reddish_count, white_count = count_reddish_and_white_sections(eye_image_path)

        print(f'Reddish Section Count: {reddish_section_count}')
        print(f'White Section Count: {white_section_count}')

        if white_section_count>=10:
            T.insert(tk.END,"Not Drugged")
        else:
            T.insert(tk.END,"Drugged")
        


# Open the image
    image = Image.open(source)

# Get pixel data
    pixels = list(image.getdata())

# Get unique colors and their counts
    unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)

# Normalize counts for better visualization
    normalized_counts = counts / counts.max()

# Create a scatter plot for each pixel
    plt.figure(figsize=(10, 6))
    plt.scatter(unique_colors[:, 0], unique_colors[:, 1], c=unique_colors / 255.0, s=normalized_counts * 500, edgecolors='none')
    plt.title('Pixel-based Chart for Input Image')
    plt.xlabel('Red')
    plt.ylabel('White')
    plt.show()


    image = cv2.imread(source)

# Calculate histogram using OpenCV
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])

# Plot the histogram
    plt.plot(hist)
    plt.title('Histogram for Image')
    plt.xlabel('Pixel Value')
    plt.ylabel('Pixel Box Count')
    plt.show()

        


root = Tk()

# specify size of window.
root.geometry("500x400")
root.config(background = "BLACK")
# Create text widget and specify size.
T = Text(root, height = 3, width = 20,background="white", foreground="red")
# Create label'
l2 = Label(root, text =" ")
l2.config(font =("Courier", 28),background = "BLACK")
l3= Label(root, text =" ")
l3.config(font =("Courier", 28),background = "BLACK")
l4 = Label(root, text =" ")
l4.config(font =("Courier", 28),background = "BLACK")
l1 = Label(root, text =" ")
l1.config(font =("Courier", 28),background = "BLACK")
l = Label(root, text = "DRUGGED EYE DETECTION")
l.config(font =("Courier", 18))



# Create button for next text.
b1 = Button(root, text = "BROWSE FOR EYE IMAGE", command = browseFiles)


# Create an Exit button.
b2 = Button(root, text = "CLICK TO CLOSE",
			command = root.destroy)

l1.pack()
l.pack()
l4.pack()
T.pack()
l2.pack()
b1.pack()
l3.pack()
b2.pack()




tk.mainloop()

