import tkinter as tk
#from tkinter import ttk, LEFT, END
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilename
import cv2
import numpy as np
global fn
import fingerprint_feature_extractor

fn=""
fileName=""

##############################################+=============================================================
root = tk.Tk()
root.configure(background="white")
# root.geometry("1300x700")


w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
root.title(" Fingerprint Verification")

# 43

# ++++++++++++++++++++++++++++++++++++++++++++
#####For background Image
image2 = Image.open('finger4.jpg')
image2 = image2.resize((w,h), Image.ANTIALIAS)

background_image = ImageTk.PhotoImage(image2)

background_label = tk.Label(root, image=background_image)
background_label.image = background_image

background_label.place(x=0, y=0)  # , relwidth=1, relheight=1)




# label_l1 = tk.Label(root,font=("Times New Roman", 30, 'bold'),
#                     background="#DC143C", fg="white", width=67, height=2)
# label_l1.place(x=0, y=0)

#
label_l2 = tk.Label(root, text="Contactless Fingerprint verification",font=("times", 30, 'bold'),
                    background="#DC143C", fg="white", width=50, height=2)
label_l2.place(x=0, y=0)



# frame_alpr = tk.LabelFrame(root, text=" Image Processing ", width=1000, height=450, bd=5, font=('times', 14, ' bold '),bg="black",fg="white")
# frame_alpr.grid(row=0, column=0, sticky='nw')
# frame_alpr.place(x=400, y=200)

    
    


def update_label(str_T):
    #clear_img()
    result_label = tk.Label(root, text=str_T, width=70, font=("bold", 25), bg='bisque2', fg='black')
    result_label.place(x=200, y=450)

###############################################################################


# def test_model():
#     global fn
#     if fn!="":
#         update_label("Model Testing Start...............")
        
#         start = time.time()
    
#         X=test_model_proc(fn)
        
#         #X1="Selected Image is {0}".format(X)
#         x2=format(X)+" Diesease is detected"
        
#         end = time.time()
            
#         ET="Execution Time: {0:.4} seconds \n".format(end-start)
        
#         msg="Image Testing Completed.."+'\n'+ x2 + '\n'+ ET
#         fn=""
#     else:
#         msg="Please Select Image For Prediction...."
        
#     update_label(msg)
    
# #############################################################################
    
def openimage():
   
    global fn, fileName
    fileName = askopenfilename(initialdir='G:\Affan College Stuff\4th Year 7th sem\BE Project\code-base', title='Select image for Analysis ',
                               filetypes=[(".png", "*.png*")])
    IMAGE_SIZE=200
    imgpath = fileName
    fn = fileName


#        img = Image.open(imgpath).convert("L")
    img = Image.open(imgpath)
    
    img = img.resize((IMAGE_SIZE,200))
    img = np.array(img)
#        img = img / 255.0
#        img = img.reshape(1,IMAGE_SIZE,IMAGE_SIZE,3)


    x1 = int(img.shape[0])
    y1 = int(img.shape[1])



    im = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(im)
    img = tk.Label(root, image=imgtk, height=250, width=250)
    img.image = imgtk
    img.place(x=300, y=150)
  
#############################################################################    

def convert_grey():
    global fn    
    IMAGE_SIZE=200
    
    img = Image.open(fn)
    img = img.resize((IMAGE_SIZE,200))
    img = np.array(img)
    
    x1 = int(img.shape[0])
    y1 = int(img.shape[1])

    gs = cv2.cvtColor(cv2.imread(fn, 1), cv2.COLOR_RGB2GRAY)

    gs = cv2.resize(gs, (x1, y1))

    retval, threshold = cv2.threshold(gs, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    print(threshold)

    im = Image.fromarray(gs)
    imgtk = ImageTk.PhotoImage(image=im)
    
    #result_label1 = tk.Label(root, image=imgtk, width=250, font=("bold", 25), bg='bisque2', fg='black',height=250)
    #result_label1.place(x=300, y=400)
    img2 = tk.Label(root, image=imgtk, height=250, width=250,bg='white')
    img2.image = imgtk
    img2.place(x=650, y=150)

    im = Image.fromarray(threshold)
    imgtk = ImageTk.PhotoImage(image=im)

    img3 = tk.Label(root, image=imgtk, height=250, width=250)
    img3.image = imgtk
    img3.place(x=970, y=150)
    #result_label1 = tk.Label(root, image=imgtk, width=250,height=250, font=("bold", 25), bg='bisque2', fg='black')
    #result_label1.place(x=300, y=400)


captured_frame = None

def capture_image():
    global captured_frame
    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        cv2.imshow('Webcam', frame)
        captured_frame = frame  # Store the captured frame

        # Wait for keypress
        key = cv2.waitKey(1)

        # if 'q' is pressed, exit the loop
        if key == ord('q'):
            break
    save_captured_image()
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def save_captured_image():
    global captured_frame
    if captured_frame is not None:
        cv2.imwrite('generated-image.jpeg', captured_frame)
        print("Captured image saved!")


def process_fingerprint():
    # global captured_frame
    global fileName 
    img = cv2.imread(fileName, 0)	
    if img is not None:
        # Convert to grayscale if necessary
        # gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Extract minutiae features
        FeaturesTerminations, FeaturesBifurcations = fingerprint_feature_extractor.extract_minutiae_features(
            img, 
            spuriousMinutiaeThresh=10, 
            invertImage=False, 
            showResult=True,  # Change to False to handle the display manually
            saveResult=False   # Change to False if you want to handle saving manually
        )

        # Example: Print the features
        # print("Terminations:", FeaturesTerminations)
        # print("Bifurcations:", FeaturesBifurcations)

        # # Example: Draw the features on the image
        # for x, y in FeaturesTerminations:
        #     cv2.circle(img, (x, y), 5, (0, 255, 0), 1)
        # for x, y in FeaturesBifurcations:
        #     cv2.circle(img, (x, y), 5, (0, 0, 255), 1)

        # # Display the image with drawn features
        # cv2.imshow('Fingerprint with Features', gray_frame)

def window():
  root.destroy()
  
  


#####################################################################################################################

button1 = tk.Button(root, text="Open Web cam", command=capture_image, width=15, height=1, font=('times', 15, ' bold '),bg="black",fg="white")
button1.place(x=50, y=350)

button2 = tk.Button(root, text=" Select Image ", command=openimage,width=15, height=1, font=('times', 15, ' bold '),bg="black",fg="white")
button2.place(x=50, y=450)

button3 = tk.Button(root, text="Image preprocess", command=convert_grey, width=15, height=1, font=('times', 15, ' bold '),bg="black",fg="white")
button3.place(x=50, y=550)

button5 = tk.Button(root, text="Process Fingerprint", command=process_fingerprint, width=15, height=1, font=('times', 15, 'bold'), bg="black", fg="white")
button5.place(x=50, y=650)


#####################################################################################################################

label_l1 = tk.Label(root, text="**  **",font=("Times New Roman", 10, 'bold'),
                    background="black", fg="white", width=250, height=2)
label_l1.place(x=0, y=800)



root.mainloop()