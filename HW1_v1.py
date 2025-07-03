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
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype('float32') / 255.0
        images_list.append(np.expand_dims(img, axis=2))    # (128, 128, 1)
    return np.array(images_list)

# 載入訓練資料（良品）及測試資料（良品與部分良品）
img_trainGood = load_img('train/good')
img_testGood = load_img('test/good')
img_testPartial = load_img('test/partial')

# 建立 Autoencoder 模型
# Autoencoder 由 Encoder 與 Decoder 組成，用於學習資料的低維表示並重建輸入
input_img = Input(shape=(IMG_SIZE, IMG_SIZE, 1))

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

# 建立模型並編譯，損失函數使用均方誤差(MSE)
autoencoder = Model(input_img, output)
autoencoder.compile(optimizer=Adam(), loss='mse')

# 訓練模型，輸入與目標皆為良品影像，讓模型學習重建良品特徵
autoencoder.fit(img_trainGood, img_trainGood, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True)

# 使用訓練好的模型對測試資料進行重建
recon_good = autoencoder.predict(img_testGood)
recon_partial = autoencoder.predict(img_testPartial)

# 計算重建誤差 (MSE) ：比較原始影像與重建影像的差異
# 理論上良品的重建誤差較低，異常品的重建誤差較高
mse_good = np.mean((img_testGood - recon_good) ** 2, axis=(1, 2, 3))
mse_partial = np.mean((img_testPartial - recon_partial) ** 2, axis=(1, 2, 3))

# 根據良品的重建誤差分佈，選擇95百分位作為判斷異常的閾值
threshold = np.percentile(mse_good, 95)

# 根據閾值將測試資料分類：誤差大於閾值視為異常(部分良品)，否則視為良品
pred_good = mse_good > threshold  # 良品預測結果應為 False (正常)
pred_partial = mse_partial > threshold  # 部分良品預測結果應為 True (異常)

# 建立真實標籤與預測標籤陣列，0代表良品，1代表部分良品
y_true = np.array([0]*len(mse_good) + [1]*len(mse_partial))
y_pred = np.concatenate([pred_good, pred_partial]).astype(int)

# 計算混淆矩陣與相關指標
TN, FP, FN, TP = confusion_matrix(y_true, y_pred).ravel()
accuracy = (TP + TN) / (TP + TN + FP + FN)  # 正確率
overkill_rate = FP / (FP + TN)  # 將良品誤判為異常的比例
underkill_rate = FN / (TP + FN)  # 將異常判為良品的比例

# 輸出評估結果
print(f"Accuracy: {accuracy:.4f}")
print(f"Overkill rate: {overkill_rate:.4f}")
print(f"Underkill rate: {underkill_rate:.4f}")