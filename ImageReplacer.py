import numpy as np
import cv2 as cv


def compute_descriptors(img):
    keypoints, descriptors = sift.detectAndCompute(img, None)
    return keypoints, descriptors


def match_img(descriptors_template, descriptors_frame):
    matches = flann.knnMatch(descriptors_template, descriptors_frame, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        pts_img = np.float32([kp_template[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts_frame = np.float32([kp_frame[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        homography_matrix, homography_mask = cv.findHomography(pts_img, pts_frame, cv.RANSAC, 5.0)
        if np.any(homography_matrix):
            return homography_matrix
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
    return None


def draw_replacement_on_frame(frame, pts_on_frame, homography_matrix,template_size):
    perspective = cv.perspectiveTransform(pts_on_frame, homography_matrix)
    perspective_matrix = cv.getPerspectiveTransform(pts_on_frame, perspective)

    # set img_sub to template location and perspective
    img_res = cv.resize(img_replacement, (img_template.shape[1], img_template.shape[0]))
    out_sub = cv.warpPerspective(img_res, perspective_matrix, (frame.shape[1], frame.shape[0]))
    mask = np.where(out_sub != [0, 0, 0])
    frame[mask] = out_sub[mask]

    # add template points to frame
    for pts in perspective:
        frame = cv.circle(frame, (int(pts[0][0]), int(pts[0][1])), 3, (0, 0, 255), 1)

    return frame


if __name__ == "__main__":
    MIN_MATCH_COUNT = 20
    FLANN_INDEX_KDTREE = 1

    cap = cv.VideoCapture(0)
    sift = cv.SIFT_create()
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)

    img_template = cv.imread('image.jpg')
    img_replacement = cv.imread('image_replacement.jpg')

    # find img_template coordinates for perspective on frame
    template_size = img_template.shape
    pts_on_frame = np.float32(
        [[0, 0], [0, template_size[0] - 1], [template_size[1] - 1, template_size[0] - 1], [template_size[1] - 1, 0]]).reshape(-1, 1, 2)

    kp_template, descriptors_template = compute_descriptors(img_template)

    while True:
        _, frame = cap.read()
        kp_frame, descriptors_frame = compute_descriptors(frame)
        homography_matrix = match_img(descriptors_template, descriptors_frame)
        if np.any(homography_matrix):
            frame = draw_replacement_on_frame(frame, pts_on_frame, homography_matrix, template_size)

        # show frame
        cv.imshow("frame", frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv.destroyAllWindows()
