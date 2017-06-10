import numpy as np
import cv2
import os
#img1 = cv2.imread('01.png',0)# queryImage
img_data = ['car_front.png', 'car_front1.png', 'car_back.png', 'car_back1.jpg', 'suv_front.png', 'suv_front1.jpg', 'suv_back.jpg', 'suv_back2.jpg']
test_data = ['1479498373962951201.jpg', '1479498377463264578.jpg', '1479498394463918193.jpg', '1479498389966519477.jpg', '1479498390964153934.jpg']
#img_name = '1479498373962951201.jpg'
#img2 = cv2.imread(img_name,0) # trainImage
#img2 = cv2.resize(img2, (440, 440))
#this can be changed
widthRatio = 440.0 / 1920.0
heightRatio = 440.0 / 1200.0
#drawMatches function on StackOverflow
def load_image_label(img_name):
    label_path = '/Users/anekisei/Documents/cs231a_project/data/label_02'
    label_matrix = np.genfromtxt(os.path.join(label_path, 'labels.csv'),delimiter=',',dtype='str')
    label_matrix = np.delete(label_matrix,[0],0)
    frame_vec = label_matrix[:,4]
    #frame_vec = frame_vec.astype(np.float)
    leftTop_x = label_matrix[:,0]
    leftTop_y = label_matrix[:,1]
    rightBot_x = label_matrix[:,2]
    rightBot_y = label_matrix[:,3]
    leftTop_x = leftTop_x.astype(np.float)
    leftTop_y = leftTop_y.astype(np.float)
    rightBot_x = rightBot_x.astype(np.float)
    rightBot_y = rightBot_y.astype(np.float)
        
    leftTop_x = leftTop_x * widthRatio
    leftTop_y = leftTop_y * heightRatio
    rightBot_x = rightBot_x * widthRatio
    rightBot_y = rightBot_y * heightRatio
    new_label = np.array([frame_vec, leftTop_x, leftTop_y, rightBot_x, rightBot_y]).T
    img_label = []
    for i in range(new_label.shape[0]):
        if new_label[i,0] == img_name:
            img_label.append([new_label[i,1], new_label[i,2], new_label[i,3], new_label[i,4]])
    img_label = np.array(img_label)
    img_label = img_label.reshape((-1,4)).astype(np.float)
    return img_label


def drawMatches(img1, kp1, img2, kp2, matches):
    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    points_detected = []
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]
    
    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')
    
    # Place the first image to the left
    out[:rows1,:cols1] = np.dstack([img1, img1, img1])
    
    # Place the next image to the right of it
    out[:rows2,cols1:] = np.dstack([img2, img2, img2])
    
    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:
        
        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx
        
        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt
        points_detected.append([x2,y2])
        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)
        
        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)
    
    
    # Show the image
    cv2.imshow('Matched Features', out)
    cv2.waitKey(0)
    cv2.destroyWindow('Matched Features')

    # Also return the image if you'd like a copy
    points_detected = np.array(points_detected)
    points_detected = points_detected.reshape((-1,2))
    return out, points_detected

total_points = 0
total_effective = 0
for m in range(len(test_data)):
    img_name = test_data[m]
    img2 = cv2.imread(img_name,0) # trainImage
    img2 = cv2.resize(img2, (440, 440))
    for n in range(len(img_data)):
        img1_name = img_data[n]
        img1 = cv2.imread(img1_name,0)
        # Initiate SIFT detector
        orb = cv2.ORB()

        # find the keypoints and descriptors with SIFT
        kp1, des1 = orb.detectAndCompute(img1,None)
        kp2, des2 = orb.detectAndCompute(img2,None)
        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match descriptors.
        matches = bf.match(des1,des2)

        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)
        # Draw first 10 matches.
        _, points_detected = drawMatches(img1,kp1,img2,kp2,matches[:30])
        N = points_detected.shape[0]
        total_points += N
        img_label = load_image_label(img_name)
        count = 0
        for i in range(N):
            x = points_detected[i,0]
            y = points_detected[i,1]
            for j in range(img_label.shape[0]):
                leftTop_x, leftTop_y, rightBot_x, rightBot_y = img_label[j,:]
                if x >= leftTop_x and x <= rightBot_x:
                    if y >= leftTop_y and y <= rightBot_y:
                        count += 1
        total_effective += count


accuracy = (1.0 * total_effective) / (1.0 * total_points)
print "total accuracy is", accuracy

#if __name__ == '__main__':
    # Load the example coordinates setup.

