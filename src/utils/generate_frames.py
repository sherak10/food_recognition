import cv2

# enter video path to read and write path to write generated video frames
read_path = ''
write_path = ''

cap = cv2.VideoCapture(read_path)
frame_index = 1

while(1):
    ret, frame = cap.read()
    if ret is False:
        break
    cv2.imshow('frame', frame)
    cv2.imwrite(write_path + 'picture' + str(frame_index) + '.png', frame)
    frame_index += 1
