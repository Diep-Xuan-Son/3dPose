# import cv2 
# import json
# import numpy as np

# path = "11.mp4"

# cap = cv2.VideoCapture(path)
# i = 0

# if (cap.isOpened() == False):
# 	print("Error opening the video file")

# while(cap.isOpened()):
# 	ret, frame = cap.read()
# 	if ret == True:
# 		cv2.imshow('{}'.format(i),frame)
# 		i += 1
# 		key = cv2.waitKey()
# 		if key == ord('q'):
# 			break

# 	else:
# 		break
# 	cv2.destroyAllWindows()

# cap.release()
# cv2.destroyAllWindows()

#content = json.load(open("label_golf.json"))
#print(content['0'])

# a = np.load("FSD_train_label.npy")
# print(a.shape)
# print(a.tolist())

a = (2.4354354)
print(a)
print(len(a))