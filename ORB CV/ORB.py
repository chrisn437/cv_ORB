import numpy as np
from scipy.signal import convolve2d
from scipy.spatial.distance import cdist

# Features from accelerated segment test
def FAST(img, N=12, threshold=0.2, nms_window=2):
    # creating a 5x5 Gaussian kernel
    kernel = np.array([[1, 4, 7, 4, 1],
                       [4, 16, 26, 16, 4],
                       [7, 26, 41, 26, 7],
                       [4, 16, 26, 16, 4],
                       [1, 4, 7, 4, 1]]) / 273

    # Convolve two 2-dimensional arrays (in this cas the image and the gaussian kernel). Same = Output has the same size as img
    img = convolve2d(img, kernel, mode='same', boundary="fill")


    # creating an array for cross checking. So we can check from pixel Ip, the pixel which is 3 pixels away to the left, right, up and down.
    cross_indices = np.array([[3,0,-3,0], [0,3,0,-3]])
    # creating an array for checking the whole circle around pixel Ip.
    circle_indices = np.array([[3,3,2,1,0,-1,-2,-3,-3,-3,-2,-1,0,1,2,3], [0,1,2,3,3,3,2,1,0,-1,-2,-3,-3,-3,-2,-1]])

    # creating an array (image) with the size of the image filled with zeros. There the corners are getting saved
    corner_img = np.zeros(img.shape)
    # creating a list where the x and y coordinated of the key points are saved
    keypoints = []
    # going through each pixel of the image to detect a corner (starting and finishing 3 pixels before the actual end, because when applying the gaussian
    # filter, these pixels are not valid
    for y in range(3, img.shape[0]-3):
        for x in range(3, img.shape[1]-3):
            # selecting the point p and assuming the intensity of this pixel (pixelvalue), we want to see if this pixel is a point of interest, or not
            Ip = img[y,x]
            # setting up a threshold which the surrounding pixels need to pass, so we can say it's a corner or not
            t = threshold*Ip if threshold < 1 else threshold
            # fast checking cross pixels only. At least 3 of the 4 pixels we are checking need to be above or below Threshold + Intensity of the current pixel.
            if np.count_nonzero(Ip+t < img[y+cross_indices[0,:], x+cross_indices[1,:]]) >= 3 or np.count_nonzero(Ip-t > img[y+cross_indices[0,:], x+cross_indices[1,:]]) >= 3:
                # If the criteria above is fullfield, then check the whole circle of 16 pixels. And if at least N (in this case 12) out of those pixels fulfill the criteria
                if np.count_nonzero(img[y+circle_indices[0,:], x+circle_indices[1,:]] >= Ip+t) >= N or np.count_nonzero(img[y+circle_indices[0,:], x+circle_indices[1,:]] <= Ip-t) >= N:
                    # Then it's a point of interest (corner), and we append it in the key point list, and insert this point in the corner_img array
                    keypoints.append([x,y])     # Note: keypoint = [col, row]
                    corner_img[y,x] = np.sum(np.abs(Ip - img[y+circle_indices[0,:], x+circle_indices[1,:]]))

    # NMS - Non Maximal Suppression to get rid of adjecent key points and multiple detections of the same object
    if nms_window != 0:
        fewer_kps = []
        # take the keypoints in the keypoint list
        for [x, y] in keypoints:
            # create a nms window (in this case we consider 3 pixels around the keypoint)
            window = corner_img[y-nms_window:y+nms_window+1, x-nms_window:x+nms_window+1]
            # taking the maximal pixelvalue out of the window and saving it to the fewer_kps list
            loc_y_x = np.unravel_index(window.argmax(), window.shape)
            x_new = x + loc_y_x[1] - nms_window
            y_new = y + loc_y_x[0] - nms_window
            new_kp = [x_new, y_new]
            if new_kp not in fewer_kps:
                fewer_kps.append(new_kp)
    else:
        fewer_kps = keypoints

    return np.array(fewer_kps)

# Binary Robust Independent Elementary Features
def BRIEF(img, keypoints, n=256, patch_size=9, sample_seed=42):
    '''
    BRIEF [Binary Robust Independent Elementary Features] keypoint/corner descriptor
    '''
    # generating a pseudo-random number generator. The fixed seed makes shure that it will produce the same results when called several times
    # This is very usefull when it comes to the "random" generated samples, because it will produce the same samples. So the same
    # pixel value comparisons will be calculated every time.
    # Like this we make sure that the binary descriptors are valid
    random = np.random.RandomState(seed=sample_seed)


    # creating a 7x7 Gaussian window
    kernel = np.array([[0, 0, 1, 2, 1, 0, 0],
                       [0, 3, 13, 22, 13, 3, 0],
                       [1, 13, 59, 97, 59, 13, 1],
                       [2, 22, 97, 159, 97, 22, 2],
                       [1, 13, 59, 97, 59, 13, 1],
                       [0, 3, 13, 22, 13, 3, 0],
                       [0, 0, 1, 2, 1, 0, 0]])/1003

    # Convolve two 2-dimensional arrays (in this cas the image and the gaussian kernel). Same = Output has the same size as img
    img = convolve2d(img, kernel, mode='same', boundary="fill")

    # creating samples. Each sample consits of random digits (but within the boundaries of the patch size)
    # We use those samples to compute the binary siganture, which will be n (in this case 512) bit long for each key point
    samples = random.randint(-(patch_size - 2) // 2 +1, (patch_size // 2), (n * 2, 2))
    samples = np.array(samples, dtype=np.int32)
    pos1, pos2 = np.split(samples, 2)

    rows, cols = img.shape

    # taking the neighbour pixels (patch) and compute a signature from this patch (in BRIEF it's a binary signature)
    mask = (  ((patch_size//2 - 1) < keypoints[:, 0])
            & (keypoints[:, 0] < (cols - patch_size//2 + 1))
            & ((patch_size//2 - 1) < keypoints[:, 1])
            & (keypoints[:, 1] < (rows - patch_size//2 + 1)))

    keypoints = np.array(keypoints[mask, :], dtype=np.intp, copy=False)
    # setting up the binary descriptor list, and filling it with zeroes/falses
    descriptors = np.zeros((keypoints.shape[0], n), dtype=bool)

    # iterating through the image and computing the binary descriptors for each patch
    for p in range(pos1.shape[0]):
        # taking x and y coordinates of two pixels within the patch
        pr0 = pos1[p, 0]
        pc0 = pos1[p, 1]
        pr1 = pos2[p, 0]
        pc1 = pos2[p, 1]
        # iterating through the key points
        for k in range(keypoints.shape[0]):
            kr = keypoints[k, 1]
            kc = keypoints[k, 0]
             # comparing the pixel intensities. If the pixel value (itensity) of the first position is smaller than the second,
             # than we return a 1 (True), otherwise we return a 0 (False)
             # And we do this for each Key point/ within the Patch, n (256) times, so we have a n (256) bit long binary descriptor for each Key Point
            if img[kr + pr0, kc + pc0] < img[kr + pr1, kc + pc1]:
                descriptors[k, p] = True

    return descriptors


def match(descriptors1, descriptors2, cross_check=True):
    # Compute distance (hamming distance) between each pair of the two collections of inputs.
    distances = cdist(descriptors1, descriptors2, metric='hamming')

    # taking the indices of the first descriptor (one of the uno cards)
    indices1 = np.arange(descriptors1.shape[0])     # [0, 1, 2, 3, 4, 5, 6, 7, ..., len(d1)] "indices of d1"
    # returning the indices of the descriptors that are the closest to the descriptor 1
    indices2 = np.argmin(distances, axis=1)

    # cross check says, if for point 1 (descritpor 1) there is a point 2 which is closest to it.
    # And now the same should be true vise versa, so if we look at point 2, the same point 1 should be the closest.
    # And this we check in cross check if it is true
    if cross_check:
        # taking the points of descriptor 1 which are closest to points of descriptor 2
        matches1 = np.argmin(distances, axis=0)

        # indices2 is the forward matches [d1 -> d2], while matches1 is the backward matches [d2 -> d1].
        mask = indices1 == matches1[indices2]
        # we are basically asking does this point in d1 matches with a point in d2 that is also matching to the same point in d1 ?
        indices1 = indices1[mask]
        indices2 = indices2[mask]



    # removing ambiguous matches.
    # ambiguous matches: matches where the closest match distance is similar to the second closest match distance
    #                   basically, the algorithm is confused about 2 points, and is not sure enough with the closest match.
    # solution: if the ratio between the distance of the closest match and that of the second closest match is more than
    #           the defined "distance_ratio", we remove this match entirely. if not, we leave it as is.

    modified_dist = distances
    # taking the indices of the minimum distances between the two descriptors
    fc = np.min(modified_dist[indices1,:], axis=1)
    modified_dist[indices1, indices2] = np.inf
    fs = np.min(modified_dist[indices1,:], axis=1)
    mask = fc/fs <= 0.5
    indices1 = indices1[mask]
    indices2 = indices2[mask]

    # sort matches using distances
    dist = distances[indices1, indices2]
    sorted_indices = dist.argsort()

    matches = np.column_stack((indices1[sorted_indices], indices2[sorted_indices]))
    return matches


if __name__ == "__main__":
    import cv2
    import os

    # defining the path where the reference images are stored
    path = 'images'
    # creating a list where the image names are stored in
    images = []
    # creating a list for the image names, without the .png .jpg ... ending
    imgNames = []
    # creating a way to access the reference images
    myList = os.listdir(path)
    print("There are", len(myList), "reference images detected")

    # iterating through the list, to read(access) all the images
    for imgs in myList:
        currentImg = cv2.imread(f'{path}/{imgs}',0)
        # adding the current image to the list
        images.append(currentImg)
        # adding the image name to the list (so the name what the image contains can be displayed)
        imgNames.append(os.path.splitext(imgs)[0])
    print(imgNames)


    # creating a list where key points are saved in
    kps1 = []
    kps2 = []
    # creating a list for descriptors
    ds1 = []
    ds2 = []

    # iterating through the images
    for img in images:

        # applying the FAST algorithm to detect points of interest (Key points/kp) for each image in the images folder
        kp1 = FAST(img, N=12, threshold=0.2, nms_window=3)

        # appending the result in the key point list 1, and multiplying it with the scale
        kps1.append(kp1)

        # finding descriptors with the BRIEF algorithm
        d1 = BRIEF(img, kp1, patch_size=9, n=256)
        # A paper also showed that a binary vector length of 512 bits can create a more valid descriptor which comes with the
        # drawback of more computational cost. That is why we used a length of 256 (had not the best pc).

        # appending the descriptors of each image to the descriptors 1 list
        ds1.append(d1)

    print("There are", len(ds1), "lists of key points and descriptors from the uno cards computed")

    cam = cv2.VideoCapture(0)

    while True:
        # access the frames of the video cam
        successful, frames = cam.read()

        # if not possible, there is an issue with reading the camera pictures
        if not successful:
            print("failed to read the cam stream")
            break

        # creating a copy of the original frames to display them later
        imgOriginal = frames.copy()


        # convert the captured frames to gray scaled
        frames = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)


        # applying the FAST algorithm to detect points of interest (Key points/kp)
        kp2 = FAST(frames, N=12, threshold=0.2, nms_window=3)
        # appending the result in the key point list 1, and multiplying it with the scale
        kps2.append(kp2)
        # finding descriptors with the BRIEF algorithm
        d2 = BRIEF(frames, kp2, patch_size=9, n=256)
        ds2.append(d2)

        # creating a list for the matches
        matchList = []
        # creating a variable, which tells us which uno card is shown
        unoCard = -1
        try:
            # iterating through the descriptors of each uno card
            for des in ds1:
                # and matching the descriptor of each uno card, against the descriptor of the current webcam frame
                matches = match(des, d2, cross_check=True)
                # Than appending the amount of matches for the 4 results in the matches list
                matchList.append(len(matches))
        except:
            pass

        # to have an overview we print how many matches were computed for each card in the frame
        print(f"{matchList} The list of matches per uno card")

        # creating a threshold, to say if this threshold isn't reached, then there is probably no uno card shown in the webcam
        threshhold = 2

        # if the matches list is not empty:
        if len(matchList) != 0:
            # and the detected matches are above the threshold
            if max(matchList) > threshhold:
                # then pick the position of the uno card which has the highest amount of matches
                unoCard = matchList.index(max(matchList))
                # and we also print out the position
                print(f"{unoCard} index of uno card")

        # if the uno card is found (value has changed from -1 to a number) than print a message onto the webcam frames
        if unoCard != -1:
            message = f"this is a {imgNames[unoCard]}"
            cv2.putText(imgOriginal, message, (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255), 2)

        # displaying the webcam frames and the message which card is shown
        cv2.imshow("Webcam", imgOriginal)
        cv2.waitKey(1)
