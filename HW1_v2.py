import os
import numpy as np
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# 參數設定：圖片大小、批次大小、訓練輪數
IMG_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 50

# 讀取資料夾中所有圖片，轉為灰階、調整大小、正規化
def load_img(load_folder):
    images_list = []
    filename_list = os.listdir(load_folder)
    for filename in sorted(filename_list):
        file_path = os.path.join(load_folder, filename)
        img = cv2.imread(file_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype('float32') / 255.0
        images_list.append(img)
    return np.array(images_list)

# 載入訓練資料（良品）及測試資料（良品與部分良品）
data_trainGood = load_img('train/good')
data_testGood = load_img('test/good')
data_testPartial = load_img('test/partial')

# 建立 Autoencoder 模型

# Autoencoder 由 Encoder 與 Decoder 組成，用於學習資料的低維表示並重建輸入
input_img = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

# Encoder 部分：將輸入影像逐步壓縮，提取重要特徵
# 第一層卷積將輸入轉為32個特徵圖，激活函數為ReLU，保持大小不變
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
# 使用最大池化降低空間維度至一半，保留重要訊息
x = MaxPooling2D((2, 2), padding='same')(x)
# 第二層卷積將特徵圖數降至16，繼續提取特徵
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
# 再次最大池化，進一步壓縮空間維度
x = MaxPooling2D((2, 2), padding='same')(x)

# Decoder 部分：將壓縮後的特徵圖逐步放大，重建輸入影像
# 卷積層將特徵圖數維持為16，激活函數為ReLU
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
# 上採樣將空間維度放大至原來的兩倍
x = UpSampling2D((2, 2))(x)
# 卷積層將特徵圖數增加至32，準備恢復更多細節
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
# 再次上採樣，回復到輸入的大小
x = UpSampling2D((2, 2))(x)
# 最後一層卷積將特徵圖數降為1，並使用sigmoid激活函數輸出0~1的像素值
output = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

# 建立模型，損失函數使用均方誤差(MSE)
keras_autoencoder = Model(input_img, output)
keras_autoencoder.compile(optimizer=Adam(), loss='mse')

# 訓練模型，輸入與目標皆為良品影像，讓模型學習重建良品特徵
keras_autoencoder.fit(data_trainGood, data_trainGood, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True)

# 使用訓練好的模型對測試資料進行重建
reconstructed_good = keras_autoencoder.predict(data_testGood)
reconstructed_partial = keras_autoencoder.predict(data_testPartial)

# 計算重建誤差 (MSE) ：比較原始影像與重建影像的差異
# 理論上良品的重建誤差較低，異常品的重建誤差較高

mse_good = []
for original, reconstructed in zip(data_testGood, reconstructed_good):
    # 計算每張圖的重建誤差（逐像素相減後平方）
    diff = (original - reconstructed) ** 2
    # 對所有 pixel 和 channel 的誤差取平均
    mse = np.mean(diff)
    # 存入結果清單
    mse_good.append(mse)
# 轉成 NumPy 陣列（和原寫法對齊）
mse_good = np.array(mse_good)

mse_partial = []
for original, reconstructed in zip(data_testPartial, reconstructed_partial):
    # 計算每張圖的重建誤差（逐像素相減後平方）
    diff = (original - reconstructed) ** 2
    # 對所有 pixel 和 channel 的誤差取平均
    mse = np.mean(diff)
    # 存入結果清單
    mse_partial.append(mse)
# 轉成 NumPy 陣列（和原寫法對齊）
mse_partial = np.array(mse_partial)

# 根據良品的重建誤差分佈，選擇95百分位作為判斷異常的閾值
threshold = np.percentile(mse_good, 95)
#threshold = 0.0006081267114495858

# 根據閾值將測試資料分類：誤差大於閾值視為異常(部分良品)，否則視為良品
pred_good = mse_good > threshold  # 良品預測結果應為 False (正常)
pred_partial = mse_partial > threshold  # 部分良品預測結果應為 True (異常)

# 建立真實標籤與預測標籤：0 = 良品，1 = 部分良品（異常）
# true_labels: 真實標籤陣列 (0=良品, 1=異常)，predicted_labels: 預測標籤陣列 (0=良品, 1=異常)
true_labels = np.array([0]*len(mse_good) + [1]*len(mse_partial))
predicted_labels = np.concatenate([pred_good, pred_partial]).astype(int)

# 計算混淆矩陣 (TN=True Negative, FP=False Positive, FN=False Negative, TP=True Positive)
TN, FP, FN, TP = confusion_matrix(true_labels, predicted_labels).ravel()

# 依據混淆矩陣計算三項指標
correct_predictions = TP + TN
total_predictions = TP + TN + FP + FN
accuracy = correct_predictions / total_predictions  # 準確率 (Accuracy): 預測正確的比例

overkill = FP / (FP + TN)  # 過殺率 Overkill rate: 良品誤判為異常的比例
underkill = FN / (TP + FN)  # 漏殺率 Underkill rate: 異常誤判為良品的比例

# 輸出評估結果（中英文說明）
print("----------------------------------------")
print("threshold: ", threshold)
print(f"Accuracy: {accuracy:.4f}")
print(f"Overkill rate: {overkill:.4f}")
print(f"Underkill rate: {underkill:.4f}")



# 合併測試圖片資料（順序與 true_labels 相同）
all_test_images = np.concatenate([data_testGood, data_testPartial])

# 找出預測錯誤的 index
error_indices = np.where(true_labels != predicted_labels)[0]

print(f"共有 {len(error_indices)} 張圖片預測錯誤：")

for idx in error_indices:
    img = all_test_images[idx]
    actual = true_labels[idx]
    pred = predicted_labels[idx]

    plt.figure()
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Actual: {actual}  Predicted: {pred}", fontsize=12, color='red')
    plt.show()


"""
# 額外視覺化：不同 threshold 對模型表現的影響
scores = np.concatenate([mse_good, mse_partial])
labels = np.array([0]*len(mse_good) + [1]*len(mse_partial))

thresholds = np.linspace(min(scores), max(scores), 100)
accuracies, overkills, underkills = [], [], []

for t in thresholds:
    preds = (scores > t).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0,1]).ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)
    ok = fp / (fp + tn) if (fp + tn) > 0 else 0
    uk = fn / (fn + tp) if (fn + tp) > 0 else 0
    accuracies.append(acc)
    overkills.append(ok)
    underkills.append(uk)

# 畫圖
plt.figure(figsize=(12, 6))
plt.plot(thresholds, accuracies, label='Accuracy', color='green')
plt.plot(thresholds, overkills, label='Overkill Rate (False Positives)', color='red')
plt.plot(thresholds, underkills, label='Underkill Rate (False Negatives)', color='blue')
plt.xlabel("Threshold")
plt.ylabel("Rate")
plt.title("Threshold vs Accuracy / Overkill / Underkill")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("threshold_performance_curve.png", dpi=300)
plt.show()
"""