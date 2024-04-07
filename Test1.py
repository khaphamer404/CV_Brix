import cv2
import numpy as np
import imutils
from imutils import contours

def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def calculate_angle_line(line1, line2):
    dx1 = line1[1][0] - line1[0][0]
    dy1 = line1[1][1] - line1[0][1]
    dx2 = line2[1][0] - line2[0][0]
    dy2 = line2[1][1] - line2[0][1]
    dot_product = dx1 * dx2 + dy1 * dy2
    length1 = calculate_distance(line1[0], line1[1])
    length2 = calculate_distance(line2[0], line2[1])
    angle_rad = np.arccos(dot_product / (length1 * length2))
    angle_deg = np.degrees(angle_rad)
    return angle_deg

# define the dictionary of digit segments so we can identify
# each digit on the thermostat
DIGITS_LOOKUP = {
	(1, 1, 1, 0, 1, 1, 1): 0,
	(0, 0, 1, 0, 0, 1, 0): 1,
	(1, 0, 1, 1, 1, 1, 0): 2,
	(1, 0, 1, 1, 0, 1, 1): 3,
	(0, 1, 1, 1, 0, 1, 0): 4,
	(1, 1, 0, 1, 0, 1, 1): 5,
	(1, 1, 0, 1, 1, 1, 1): 6,
	(1, 0, 1, 0, 0, 1, 0): 7,
	(1, 1, 1, 1, 1, 1, 1): 8,
	(1, 1, 1, 1, 0, 1, 1): 9
}

ratio_rs=5
image = cv2.imread('Image1/Img (40).jpg',0)
template = cv2.imread('Template/Template_rs.jpg',0)

sift = cv2.SIFT_create()
bf = cv2.BFMatcher()

# Tìm key points và descriptors của template và hình ảnh
keypoints_template, descriptors_template = sift.detectAndCompute(template, None)
keypoints_image, descriptors_image = sift.detectAndCompute(image, None)

matches = bf.knnMatch(descriptors_template, descriptors_image, k=2)

# Áp dụng RANSAC để loại bỏ các điểm không chính xác
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)
len_good_matches=len(good_matches)

if len_good_matches > 5:
     # Lấy key points từ các điểm khớp
    src_pts = np.float32([keypoints_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_image[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Tính toán ma trận chuyển đổi sử dụng hàm findHomography và RANSAC
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    h_t, w_t = template.shape
    pts = np.float32([[0, 0], [0, h_t-1], [w_t-1, h_t-1], [w_t-1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)
    points_template = np.squeeze(dst)

    # Calculate distances between points
    distances_template = [calculate_distance(points_template[i], points_template[(i+1) % 4]) for i in range(4)]
    # Calculate angles between lines
    angles_template = [calculate_angle_line((points_template[i], points_template[(i+1) % 4]), (points_template[(i+1) % 4], points_template[(i+2) % 4])) for i in range(4)]
    # Check conditions
    template_check = all(70 < angle < 110 for angle in angles_template) and all(d >= 70 for d in distances_template)

    if (template_check):
        dst_final=dst
        M_inverse=np.linalg.inv(M)
        warped_image = cv2.warpPerspective(image, M_inverse, (w_t,h_t))
    else:
        print('Ko tim thay')

# Đặc điểm trong ảnh gốc
pts_source_skew = np.float32([[190, 105], [185, 180], [228, 105]])
# Đặc điểm tương ứng trong ảnh đích (nghiêng)
pts_destination_skew = np.float32([[188, 105], [188, 180], [226, 105]])
# Tính ma trận chuyển đổi affine
matrix_skew = cv2.getAffineTransform(pts_source_skew, pts_destination_skew)
skewed_image = cv2.warpAffine(warped_image, matrix_skew, (warped_image.shape[1], warped_image.shape[0]))

radius = 3
color = (0, 0, 255)
thickness = -1

numberzone_Image = skewed_image[95:178, 120:310]
_, binary_numberzone_Image = cv2.threshold(numberzone_Image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Xác định kernel hình chữ nhật với chiều ngang lớn hơn chiều dọc
kernel_erode = np.ones((3, 1), np.uint8)
kernel_dilate = np.ones((5, 1), np.uint8)
# Thực hiện phép erosion
eroded_image = cv2.erode(binary_numberzone_Image, kernel_erode, iterations=1)
# Thực hiện phép dilation
dilated_image = cv2.dilate(eroded_image, kernel_dilate, iterations=1)

cnts = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
digitCnts = []

# Vẽ tất cả contours lên hình ảnh gốc
# cv2.drawContours(numberzone_Image, cnts, -1, (0, 255, 0), 2)

# cv2.imshow("Contours", dilated_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
    
for c in cnts:
	(x, y, w, h) = cv2.boundingRect(c)
	if (w >= 30 or w <20) and w <= 100 and (h >= 50 and h <= 100):
		digitCnts.append(c)

# sort the contours from left-to-right, then initialize the
# actual digits themselves
digitCnts = contours.sort_contours(digitCnts, method="left-to-right")[0]
digits = []

# loop over each of the digits
for c in digitCnts:
    # extract the digit ROI
    (x, y, w, h) = cv2.boundingRect(c)
    roi = binary_numberzone_Image[y:y + h, x:x + w]
    (roiH, roiW) = roi.shape
    (dW, dH) = (int(roiW * 0.25), int(roiH * 0.15))
    dHC = int(roiH * 0.05)
	# define the set of 7 segments
    segments = [
	    ((0, 0), (w, dH)),	# top
	    ((0, 0), (dW, h // 2)),	# top-left
	    ((w - dW, 0), (w, h // 2)),	# top-right
	    ((0, (h // 2) - dHC) , (w, (h // 2) + dHC)), # center
	    ((0, h // 2), (dW, h)),	# bottom-left
	    ((w - dW, h // 2), (w, h)),	# bottom-right
	    ((0, h - dH), (w, h))	# bottom
    ]

    on = [0] * len(segments)
    
    for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
        segROI = roi[yA:yB, xA:xB]
        total = cv2.countNonZero(segROI)
        area = (xB - xA) * (yB - yA)
        if total / float(area) > 0.5:
            on[i]= 1
    digit = DIGITS_LOOKUP[tuple(on)]
    digits.append(digit)
    cv2.rectangle(numberzone_Image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(numberzone_Image, str(digit), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

# Hiển thị hình ảnh sau khi vẽ contours
print(digits)
cv2.imshow("Output", numberzone_Image)
cv2.waitKey(0)
cv2.destroyAllWindows()
