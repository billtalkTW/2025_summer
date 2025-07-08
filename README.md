 (cd "$(git rev-parse --show-toplevel)" && git apply --3way <<'EOF' 
diff --git a//dev/null b/README.md
index 0000000000000000000000000000000000000000..efdf06949bd2bbae52c1c655b7c07f2c72646216 100644
--- a//dev/null
+++ b/README.md
@@ -0,0 +1,53 @@
+# 專案簡介
+
+本專案包含三個不同版本的程式，用於分析影像資料並區分良品與部分良品。程式碼分別為 `HW1_v1.py`、`HW1_v2.py`、`HW1_v3.py`。
+
+## 程式碼差異
+
+- **HW1_v1.py**：
+  - 使用 Keras 建立灰階影像的 autoencoder。
+  - 透過重建誤差 (MSE) 判斷測試影像是否為異常。
+  - 最後輸出 Accuracy、Overkill rate 及 Underkill rate。
+
+- **HW1_v2.py**：
+  - 讀取彩色影像 (3 channels) 並訓練 autoencoder。
+  - 進一步計算不同 threshold 對模型表現的影響，並能畫出曲線圖。
+  - 輸出與 v1 相同的指標，但提供更多視覺化與分析功能。
+
+- **HW1_v3.py**：
+  - 不使用深度學習，改以影像處理方式偵測黑色圓形面積。
+  - 依面積是否小於門檻值來判斷影像狀態。
+  - 同樣輸出 Accuracy、Overkill rate 及 Underkill rate，並顯示誤判圖片。
+
+## 資料夾結構
+
+```
+train/
+    good/       # 用於訓練的良品影像
+
+test/
+    good/       # 測試用的良品影像
+    partial/    # 測試用的部分良品影像
+```
+
+請將影像依照以上目錄放置，以便程式載入對應的資料集。
+
+## 執行方式
+
+在專案根目錄下，可分別執行下列指令：
+
+```bash
+python HW1_v1.py
+python HW1_v2.py
+python HW1_v3.py
+```
+
+程式完成後會輸出類似下列結果（以 HW1_v1.py 為例）：
+
+```
+Accuracy: 0.95
+Overkill rate: 0.02
+Underkill rate: 0.05
+```
+
+實際數值會依資料集而異。
 
EOF
)
