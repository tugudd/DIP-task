import cv2

#the object detection enhancement on exercise video explained in report
def checkExercise(y):
    if y >= 300 and y <= 500:
        return True
    
    return False

#the object detection enhancement on office video explained in report
def checkOffice(x, frameCnt):
    if x >= 640 and x <= 900:
        return True
    
    if frameCnt <= 44 and h <= 70:
        return True
    
    if frameCnt > 160 and frameCnt <= 290 and x <= 450 and h <= 200:
        return True
    
    return False

#a separate blur function to avoid writing same things multiple times
def blur(x, y, w, h, frame):
    region = frame[y : y+h, x : x+w, :]
    region = cv2.blur(region, (int (h / 5), int (h / 5)), cv2.BORDER_CONSTANT)
    return region
    

#putting watermarks in an array to alternate between them
watermarks = [cv2.imread("watermark1.png"), cv2.imread("watermark2.png")] 

face_cascade = cv2.CascadeClassifier("face_detector.xml")

street = cv2.VideoCapture('street.mp4')
streetOut = cv2.VideoWriter('processed_street.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30.0, (1280,720))
exercise = cv2.VideoCapture('exercise.mp4')
exerciseOut = cv2.VideoWriter('processed_exercise.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30.0, (1920, 1080))
office = cv2.VideoCapture('office.mp4')
officeOut = cv2.VideoWriter('processed_office.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30.0, (1920, 1080))

#retrieving the number of frames for each video
frameCntStreet = int (street.get(cv2.CAP_PROP_FRAME_COUNT))
frameCntExercise = int (exercise.get(cv2.CAP_PROP_FRAME_COUNT))
frameCntOffice = int (office.get(cv2.CAP_PROP_FRAME_COUNT))

#putting videos in an array to loop through them one by one
videos = [street, exercise, office]
for video in videos:
    frameCnt = 0 #denotes the number of the current frame
    talking = cv2.VideoCapture('talking.mp4')
    watermarkIndex = 0 #index in watermarks array, can be either 0 or 1
    
    #if all frames are processed, the while loop will break
    while True:
        success, frame = video.read()
        if success:
            frameCnt += 1
            faces = face_cascade.detectMultiScale(frame, 1.3, 5)
            
            for (x, y, w, h) in faces:
                if video == street:
                    frame[y : y+h, x : x+w, :] = blur(x, y, w, h, frame)
                    
                #ifs are used for applying enhancements on exercise and office video
                elif video == exercise and checkExercise(y):
                    frame[y : y+h, x : x+w, :] = blur(x, y, w, h, frame)
                
                elif video == office and checkOffice(x, frameCnt):
                    frame[y : y+h, x : x+w, :] = blur(x, y, w, h, frame)
                
            success_talking, frame_talking = talking.read()
            if success_talking:    
                [height, width, layers] = frame_talking.shape
                
                #rescaling to 30%
                height = int (height * 0.3)
                width = int (width * 0.3)
                
                resized = cv2.resize(frame_talking, (width, height))
                resized = cv2.copyMakeBorder(resized, 7, 7, 7, 7, cv2.BORDER_CONSTANT) # adding borders
                frame[50 : 64 + height, 50 : 64 + width, :] = resized # overlaying the talking video
            
            #street video has equal resolution with watermarks
            if video == street:
                frame = cv2.addWeighted(frame, 1.0, watermarks[watermarkIndex], 0.6, 0)
            
            #watermarks need to be resized to fit exercise and office videos
            else:
                frame = cv2.addWeighted(frame, 1.0, cv2.resize(watermarks[watermarkIndex], (1920, 1080)), 0.6, 0)
            
            #if 90 frames are processed, the current watermark needs to change
            if frameCnt % 90 == 0 and frameCnt != 0:
                watermarkIndex = 1 - watermarkIndex
            
            #this section used for writing and diplaying which frame is being processed 
            if video == street:
                streetOut.write(frame)
                print ("street.mp4: ", frameCnt, " out of ", frameCntStreet, " frames")
            elif video == exercise:
                exerciseOut.write(frame)
                print ("exercise.mp4: ", frameCnt, " out of ", frameCntExercise, " frames")
            else: 
                officeOut.write(frame)
                print ("office.mp4: ", frameCnt, " out of ", frameCntOffice, " frames")
            
            
            # cv2.imshow('The Video', frame)  
            # if cv2.waitKey(1) == ord ('q'):
            #     break
        
        else:
            break
    
    video.release()

cv2.destroyAllWindows()  


