import argparse
import os
import torch, torchvision
import cv2
import time

import numpy as np


# Fixes issue with 'torch.cat' (used somewhere when calling forward functions)
import ctypes
ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll')



from net import get_model

__preprocess_transform__ = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor()
    ])

def preprocess_image(img):
    # applies transformation and adds a 'fake' batch dimension (.unsqueeze(0))
    transform = __preprocess_transform__
    return transform(img).unsqueeze(0)

def predict(img, model, timestamp=False):
    """ Returns a prediction as a string for the specified model and input. 
    If timestamp is true, returns a tuple (prediction, prediction_time)
    """
    img = preprocess_image(img).to(next(model.parameters()).device)
    dt = time.perf_counter()
    yp = model(img)
    dt = time.perf_counter() - dt
    dt *= 1000
    yp = torch.sum(yp.softmax(1) * torch.arange(yp.shape[-1]).to(yp.device), dim=1).item()
    if timestamp:
        return str(round(yp, 2)), dt
    else:
        return str(round(yp, 2))


def get_model_fname(dirname):
    fname = [el for el in next(os.walk(dirname))[2] if el.endswith('.pth')]
    if len(fname) > 1:
        raise RuntimeError('cannot decide within multiple models in {}'.format(dir))
    return fname[0]


def prepare_model(model, path, classes, file_name=None):
    """ Loads model from parent directory path. Models are saved in a way that can be handled with Pytorch 1.4 (before 'zip serialization' of Pytorch 1.5)."""
    if file_name == None:
        file_name = get_model_fname(path)
    path = os.path.join(path, file_name)

    state_dict = torch.load(path)
    model = get_model(model, classes)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def overlay_transparent(background, overlay, x=0, y=0):
    """ Merges the frame or image (background) with the transparent layer on which rectangles and labels are drawn. 
    The parameters (x,y) have been used to verify the position of the overlay."""

    background_width = background.shape[1]
    background_height = background.shape[0]

    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype = overlay.dtype) * 255
            ],
            axis = 2,
        )

    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_image

    return background



    
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='2classifiermn', help="Type of model to use in the demo. Default: '2classifiermn'.", choices=['mobilenet', '2classifiermn', 'resnet18'])
parser.add_argument('--classes', type=int, default=117, help='Number of classes for the model (must be the same of the loaded model). Default: 117.')
parser.add_argument('--model_file_name', type=str, help='Filename for model retrieval; to be intended as relative name inside specified path. Default: any file with .pth extension.')
parser.add_argument('--file', type=str, help='File to perform age estimation. If not specified, usage is in realtime with integrated webcam. Accepted formats: images (.png, .jpg); videos (.mp4)')
parser.add_argument('--result_path', type=str)
parser.add_argument('--face_window_restriction', type=float, help='Threshold to perform age estimation, expressed as fraction of input data width. Default: 0.25 for images, 0.083 for videos.')
parser.add_argument('--path', type=str, default=".\\model\\", help='Path from which will be loaded the model.')



def demo_video(value, args):
    """ Applies the model from the directory specified in 'args' to a video. 
    The parameter 'value' handles real-time application: when put to 0 the execution is real-time, while if it is a string then it is interpreted as the path of the video to analyze."""
    
    capturer = cv2.VideoCapture(value)
    output = None

    # Gets frame width, height and frames-per-second
    fw = int(capturer.get(3))
    fh = int(capturer.get(4))
    fps = round(capturer.get(cv2.CAP_PROP_FPS))

    if args.face_window_restriction != None:
        min_fw = fw*args.face_window_restriction 
    
    if value != 0:
        res_dir = os.path.join('demo_videos', args.path)
        if not os.path.isdir(res_dir):
            os.mkdir(os.path.join(res_dir))
        output = cv2.VideoWriter(os.path.join(res_dir, "aE_" + os.path.basename(value)), cv2.VideoWriter_fourcc(*'mp4v'), fps, (fw, fh))
        print("Processing video data...")
    else:
        print("Press ESC to terminate realtime demo.")
    
    lab = '0'

    # Number of frames between evaluations
    hop = round(fps * 0.25)

    curr_frame = 0

    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    model = prepare_model(args.model, args.path, args.classes, args.model_file_name)

    faces_layer = None

    # Restriction on size of analyzed faces
    wres = 1/12

    if args.face_window_restriction != None:
        if not (args.face_window_restriction >= 1.0 or args.face_window_restriction <= 0.0): 
            wres = args.face_window_restriction 
        else:
            print("Warning: bad window threshold value detected. Setting to default value.")

    while True:
        ret, frame = capturer.read()
        
        if not ret:
            break
        
        if curr_frame % hop == 0:
            # Used to draw rectangles and labels correctly
            faces_layer = np.zeros((fh, fw, 4))


            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.1)
            toclassify = None            
            for x, y, w, h in faces:

                # Slightly enlarges the rectangle of analysis
                x = x-10
                y = y-10
                w = w + 10*2
                h = h + 10*2


                    
                if w / fw > wres:   
                    toclassify = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    toclassify = toclassify[y:y+h, x:x+w]
                    
                    try:
                        lab = predict(toclassify, model)
                        cv2.rectangle(faces_layer, pt1=(x,y), pt2=(x + w, y + h), color=(0, 255, 0, 255), thickness=2)
                        cv2.putText(faces_layer, "Age:" + lab, (x, y+h+25), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0, 255))
                    except ValueError:
                        pass

        if faces_layer is not None:
            frame = overlay_transparent(frame, faces_layer)

        if value == 0:
            # Shows in real time    
            cv2.imshow('AgeEstimation', frame)
        else:
            output.write(frame)
        key_pressed = cv2.waitKey(30)

        # If escape key is pressed then exits real-time demo
        if key_pressed == 27 and value == 0:
            break
        
        curr_frame += 1

    capturer.release()
    if output != None:
        output.release()
    cv2.destroyAllWindows()


def demo_image(args):
        face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        model = prepare_model(args.model, args.path, args.classes, args.model_file_name)

        img = cv2.imread(args.file)
        fw = img.shape[0]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.1)

        # Restriction on size of analyzed faces
        wres = 1/4
        if args.face_window_restriction != None:
            wres = args.face_window_restriction 

        for (x, y, w, h) in faces:

            # Slightly enlarges the rectangle of analysis
            x = x-10
            y = y-10
            w = w + 10*2
            h = h + 10*2
        
            if w / fw > wres:
                cv2.rectangle(img, (x, y), (x+w, y+h), color=(0, 255, 0), thickness=2)
                toclassify = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                toclassify = toclassify[y:y+h, x:x+w]
                lab = predict(toclassify, model)
                cv2.putText(img, "Age: " + lab, (x, y-10), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0))
        
        cv2.imshow('AgeEstimation', img)
        print("Press any key to exit demo. File will be saved.")
        cv2.waitKey()

        res_dir = os.path.join('demo_images', args.path)
        if not os.path.isdir(res_dir):
            os.mkdir(os.path.join(res_dir))
        fname = 'aE_' + os.path.basename(args.file)
        cv2.imwrite(os.path.join(res_dir, fname), img)



def main():

    args = parser.parse_args()
    
    if args.file is None:
        demo_video(0, args)
    elif args.file.endswith(('.png', '.jpg')):
        demo_image(args)
    elif args.file.endswith('.mp4'):
        demo_video(args.file, args)
    else: 
        raise ValueError("cannot open file '{}'".format(args.file))

if __name__ == '__main__':
    main()
