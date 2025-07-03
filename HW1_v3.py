import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# 計算圖像中黑色圓(最大輪廓)的面積
def get_black_circle_area(image_path): 
    gray_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 將灰階影像轉換成黑白黑二值影像，並反轉黑與白
    _, binary_img = cv2.threshold(gray_img, 70, 255, cv2.THRESH_BINARY_INV)

    # 尋找所有“最外層輪廓”，且“只保留關鍵點(壓縮直線上冗餘點)”
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return 0, gray_img

    # 找出面積最大的輪廓
    largest_contourArea = max(contours, key=cv2.contourArea)

    # 得到其面積並畫出輪廓
    area = cv2.contourArea(largest_contourArea)
    cv2.drawContours(gray_img, [largest_contourArea], -1, (0, 255, 0), 1)

    return area, gray_img

# 載入訓練資料夾（good）
train_folder = "train/good"
black_areas = []
for filename in sorted(os.listdir(train_folder)):
    file_path = os.path.join(train_folder, filename)
    black_circle_area, marked_img = get_black_circle_area(file_path)
    if black_circle_area > 0:
        black_areas.append(black_circle_area)
# 計算平均與標準差
mean_area = np.mean(black_areas)
print(f"面積平均: {mean_area:.2f}")
print("----------------------------------------")
# 面積門檻
area_threshold = mean_area * 0.958

#  載入測試資料夾（good）
test_good_folder = "test/good"
test_good_img_count, test_good_pred_good_count, test_good_pred_partial_count = 0, 0, 0
mismatched_images = []
for filename in sorted(os.listdir(test_good_folder)):
    file_path = os.path.join(test_good_folder, filename)
    area, img = get_black_circle_area(file_path)
    test_good_img_count += 1
    # 根據面積是否大於面積門檻來判斷是否遮蔽
    if area < area_threshold:
        test_good_pred_partial_count += 1
        mismatched_images.append((file_path, "GOOD", "PARTIAL"))
    else:
        test_good_pred_good_count +=1
print("----------------------------------------")
print("total_test_good_img = ", test_good_img_count)
print("test_good_pred_good_count = ", test_good_pred_good_count)
print("test_good_pred_partial_count = ", test_good_pred_partial_count)

# 載入測試資料夾（partial）
test_partial_folder = "test/partial"
test_partial_img_count, test_partial_pred_good_count, test_partial_pred_partial_count = 0, 0, 0
for filename in sorted(os.listdir(test_partial_folder)):
    file_path = os.path.join(test_partial_folder, filename)
    area, img = get_black_circle_area(file_path)
    test_partial_img_count += 1
    # 根據面積是否大於面積門檻來判斷是否遮蔽
    if area < area_threshold:
        test_partial_pred_partial_count += 1
    else:
        test_partial_pred_good_count +=1
        mismatched_images.append((file_path, "PARTIAL", "GOOD"))
print("----------------------------------------")
print("total_test_partial_img = ", test_partial_img_count)
print("test_partial_pred_good_count = ", test_partial_pred_good_count)
print("test_partial_pred_partial_count = ", test_partial_pred_partial_count)

# 計算指標
accuracy = (test_good_pred_good_count + test_partial_pred_partial_count) / (test_partial_img_count + test_good_img_count)
overkill = test_good_pred_partial_count / test_good_img_count
underkill = test_partial_pred_good_count / test_partial_img_count
print("----------------------------------------")
print(f"Accuracy: {accuracy:.4f}")
print(f"Overkill rate: {overkill:.4f}")
print(f"Underkill rate: {underkill:.4f}")
print(f"共有 { test_good_pred_partial_count + test_partial_pred_good_count} 張圖片預測錯誤")

for file_path, actual_label, pred_label in mismatched_images:
    area, img = get_black_circle_area(file_path)
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    if actual_label == "GOOD":
        plt.text(0.5, 1.07, f"Actual: {actual_label}", fontsize=12, color='lime', ha='center', transform=plt.gca().transAxes)
        plt.text(0.5, 1.01, f"Pred: {pred_label}", fontsize=12, color='red', ha='center', transform=plt.gca().transAxes)
    elif actual_label == "PARTIAL":
        plt.text(0.5, 1.07, f"Actual: {actual_label}", fontsize=12, color='red', ha='center', transform=plt.gca().transAxes)
        plt.text(0.5, 1.01, f"Pred: {pred_label}", fontsize=12, color='lime', ha='center', transform=plt.gca().transAxes)
    plt.show()
