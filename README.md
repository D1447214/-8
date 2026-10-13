# 練習 8｜人臉偵測實驗報告

這個小工具可載入單張圖片或從攝影機擷取一張畫面，使用 OpenCV Haar cascade 偵測人臉、畫框並計數，接著比較三組參數。它只偵測人臉位置，不辨識個人身分。

## 執行方式

使用 Windows PowerShell，在此資料夾執行以下指令。這台電腦的 Python 3.12 安裝位置如下；若 Python 安裝在其他位置，請替換第一行的路徑：

```powershell
& "C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe" -m venv .venv312
.\.venv312\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python .\face_detection_lab.py
```

若 PowerShell 不允許啟用環境，可不啟用，改用完整路徑執行：

```powershell
.\.venv312\Scripts\python.exe -m pip install -r requirements.txt
.\.venv312\Scripts\python.exe .\face_detection_lab.py
```

選擇圖片或擷取攝影機畫面，輸入影像中的真實人數，選擇測試情境（正面、側臉、背光、遮擋），再按「比較三組參數」。情境選單只記錄條件，不會改變照片；選擇情境時，光線、姿勢、遮擋欄位會自動配合更新。

要觀察不同情境的差異，請分別載入符合情境的實拍照片，再選擇對應的測試情境並執行比較。建議使用同一組人拍攝的照片；單純更改情境選單不會令偵測結果改變。

## 參數與輸出

| 組別 | scaleFactor | minNeighbors | minSize | 用途 |
|---|---:|---:|---:|---|
| A_敏感 | 1.03 | 2 | 影像短邊的 6%（至少 16 px） | 較容易偵測到小臉，也可能增加誤判 |
| B_平衡 | 1.08 | 3 | 影像短邊的 8%（至少 16 px） | 中間設定 |
| C_保守 | 1.12 | 4 | 影像短邊的 10%（至少 16 px） | 減少較弱的候選框，也可能漏掉人臉 |

結果存放在 `實驗結果`：每組一張標註圖片，以及追加寫入的 `實驗紀錄.csv`。CSV 記錄情境、參數、偵測框數、TP／FP／FN、耗時和觀察欄位。介面會顯示 Precision = TP / (TP + FP) 與 Recall = TP / (TP + FN)；分母為 0 時顯示「無法計算」。

**TP／FP／FN 的解讀：**程式用「預測人數與手動輸入真實人數的差」計算計數層級指標：TP = min(預測數, 真實數)、FP = max(預測數−真實數, 0)、FN = max(真實數−預測數, 0)。這是計數比較，不是逐個框以 IoU 配對的物件偵測指標。

## 環境需求

- Python 3.10 或更新版本
- OpenCV 4.x（本程式使用 Haar cascade；OpenCV 5 已調整此 API）
- 可用的攝影機（僅使用「攝影機擷取一張」時需要）
