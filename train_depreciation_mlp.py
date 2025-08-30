import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from datetime import datetime
import os

import torch.optim as optim  
from torch.optim.lr_scheduler import ReduceLROnPlateau 
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler  
from sklearn.utils.class_weight import compute_class_weight  
import pickle  
def train_mlp_from_csv(csv_path="depreciation_records.csv", output_path="depreciation_mlp.pth"):
    """
    從 CSV 資料訓練或微調折舊分析用的 MLP 模型，並儲存為 .pth 檔案。
    若模型已存在，則載入並繼續訓練；否則新建模型。
    """

    # 📥 讀取資料
    df = pd.read_csv(csv_path)
    X = torch.tensor(df[["defect_index", "avg_depth", "max_depth", "total_area"]].values, dtype=torch.float32)
    y = torch.tensor(df["grade"].map({"A - normal": 0, "B - Under_observation": 1, "C - Recommended_repair": 2}).values, dtype=torch.long)

    # 📦 建立資料集與 DataLoader
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    # 🧠 模型建立或載入
    model = DepreciationMLP()
    if os.path.exists(output_path):
        print(f"📂 偵測到已存在模型 {output_path}，將載入並繼續訓練")
        model.load_state_dict(torch.load(output_path,weights_only=True))  # 載入權重

    # ⚙️ 訓練元件設定
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # 🔁 模型訓練迴圈
    for epoch in range(50):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

    # 💾 儲存模型
    # torch.save(model, output_path)
    torch.save(model.state_dict(), output_path)  # 只儲存權重
    print(f"✅ 模型已訓練並儲存至 {output_path}")

def compute_depreciation_metrics(defects):
    """
    📌 用途：根據缺陷清單計算折舊分析的核心指標(defect_index)。
    每個缺陷的影響力以「面積 × 深度」表示，並統計整體缺陷數量、平均深度、最大深度與總面積。
    🔢 回傳內容：
    - defect_index：折舊指數(面積 × 深度 的加總)
    - defect_count：缺陷數量
    - avg_depth：平均深度
    - max_depth：最大深度
    - total_area：所有缺陷的總面積
    """
    if not defects:
        return {
            "defect_index": 0,
            "defect_count": 0,
            "avg_depth": 0,
            "max_depth": 0,
            "total_area": 0
        }

    defect_index = sum([d['area'] * d['depth'] for d in defects])
    avg_depth = np.mean([d['depth'] for d in defects])
    max_depth = np.max([d['depth'] for d in defects])
    total_area = sum([d['area'] for d in defects])
    return {
        "defect_index": defect_index,
        "defect_count": len(defects),
        "avg_depth": avg_depth,
        "max_depth": max_depth,
        "total_area": total_area
    }

def classify_depreciation(defect_index):
    """
    📌 用途：根據折舊指數(defect_index)進行分級判斷，協助使用者快速了解物件目前的健康狀態。
    📊 分級邏輯：
    - A：normal(defect_index < 50)
    - B：Under_observation(50 ≤ defect_index < 150)
    - C：Recommended_repair(defect_index ≥ 150)
    🔁 回傳：
    對應的折舊等級字串(含建議)
    """
    """基於你數據的25%和75%百分位數，能確保：
        約25%的數據為A級(normal)
        約50%的數據為B級(Under_observation)
        約25%的數據為C級(Recommended_repair)
        其他考慮的方案：
        標準差法：A<2971, B<6225, C≥6225
        保守法：A<4631, B<6741, C≥6741 """
    if defect_index < 3876:
        return "A - normal"
    elif defect_index < 5554:
        return "B - Under_observation"
    else:
        return "C - Recommended_repair"

def generate_depreciation_record(defects,mlp_model=None):
    """
    📌 用途：整合折舊分析流程，生成一筆完整的紀錄。
    包含分析時間、折舊等級、各項指標與原始缺陷清單，方便儲存、追蹤與可視化。
    🧩 組成：
    - timestamp：分析時間(格式：YYYY-MM-DD HH:MM)
    - grade：折舊等級(由 classify_depreciation 判斷)
    - defect_index / defect_count / avg_depth / max_depth / total_area：由 compute_depreciation_metrics 計算
    - defects：原始缺陷清單(含面積、深度、位置等)
    🔁 回傳：
    一個 dict 結構的折舊分析紀錄
    """
    metrics = compute_depreciation_metrics(defects)
    if mlp_model: #使用 MLP 模型根據缺陷指標預測折舊等級。
        grade, confidence = classify_depreciation_mlp(metrics, mlp_model)
    else:       #使用簡單的閾值分類
        grade = classify_depreciation(metrics["defect_index"])
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    return {
        "timestamp": timestamp,
        "grade": grade,
        "confidence": confidence if mlp_model else "N/A",
        **metrics,
        "defects": defects
    }

import torch.nn as nn
class DepreciationMLP(nn.Module):
    """ 折舊分析用的多層感知器(MLP)模型
    Linear 1 -
    接收折舊指標: defect_index、avg_depth、max_depth、total_area 四個輸入特徵，
    預測折舊等級(A、B、C 三類)。
    輸入維度:4(defect_index、avg_depth、max_depth、total_area)
    隱藏層維度:16
    激活函數:ReLU 非線性轉換
    Linear 2 -
    輸出維度:3(對應三個折舊等級類別 A/B/C)
    🔁 前向傳播流程:
    輸入 → 線性層1 → ReLU → 線性層2 → 輸出 logits
    """
    def __init__(self, input_dim=4, hidden_dim=16, output_dim=3):
        super().__init__()# 初始化 nn.Module 父類別
        # 🧠 定義 MLP 架構：輸入層 → 隱藏層 → 輸出層
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),# 線性層：輸入維度 → 隱藏層維度
            nn.ReLU(),# 啟用函數：ReLU 非線性轉換
            nn.Linear(hidden_dim, output_dim)# 線性層：隱藏層 → 輸出類別數
        )
    def forward(self, x):
        return self.model(x)# 前向傳播：將輸入 x 傳入模型並回傳輸出

def classify_depreciation_mlp(metrics, mlp_model):
    """
    使用 MLP 模型根據缺陷指標預測折舊等級，並回傳信心分數。
    🔢 輸入：metrics dict(包含 defect_index、avg_depth、max_depth、total_area)
    🔁 回傳：tuple → (折舊等級字串, 信心分數)
    """
    # 🧮 將指標轉為張量並加上 batch 維度
    input_tensor = torch.tensor([
        metrics["defect_index"],
        metrics["avg_depth"],
        metrics["max_depth"],
        metrics["total_area"]
    ], dtype=torch.float32).unsqueeze(0)

    # 🚫 推論模式(停用梯度)
    with torch.no_grad():
        logits = mlp_model(input_tensor)  # 前向傳播,logits 是 MLP 模型的原始輸出(未經標準化)，通常是每個類別的分數
        probs = torch.softmax(logits, dim=1)  # 計算 softmax 機率分布，softmax 將 logits 轉換為機率分布，使所有類別的機率加總為 1。
        #EX:logits = [1.2, 3.5, 0.8] (所有類別的機率加總為1時)-> probs = [0.12, 0.82, 0.06]
        pred = torch.argmax(probs, dim=1).item()  # 取得預測類別索引，pred 是最大機率的類別索引(即模型預測的分類)。
        confidence = probs[0, pred].item()  # 取得該類別的信心分數，是該類別的機率值，代表模型對這個預測的信心。

    # 📤 回傳等級與信心分數
    label = ["A - normal", "B - Under_observation", "C - Recommended_repair"]
    # 信心分數(confidence)的用途：
    # 報表呈現：讓使用者知道預測是否可靠
    # 警示機制：若信心低於某門檻(例如 0.6)，可標示為「不確定」
    # 模型評估：可用於 ROC 曲線、Precision-Recall 分析
    # 決策支援：高信心可自動通過，低信心可轉人工複核
    return label[pred], confidence

class EnhancedDepreciationMLP(nn.Module):  
    """改良版折舊分析 MLP 模型  
    - 更深的網路架構（3層隱藏層）  
    - Dropout 防過擬合  
    - BatchNorm 穩定訓練  
    - 支援更多特徵輸入  
    """  
    def __init__(self, input_dim=8, hidden_dims=[64, 32, 16], output_dim=3, dropout_rate=0.3):  
        super().__init__()  
          
        layers = []  
        prev_dim = input_dim  
          
        # 建立多層隱藏層  
        for hidden_dim in hidden_dims:  
            layers.extend([  
                nn.Linear(prev_dim, hidden_dim),  
                nn.BatchNorm1d(hidden_dim),  
                nn.LeakyReLU(0.1),  
                nn.Dropout(dropout_rate)  
            ])  
            prev_dim = hidden_dim  
          
        # 輸出層  
        layers.append(nn.Linear(prev_dim, output_dim))  
          
        self.model = nn.Sequential(*layers)  
          
    def forward(self, x):  
        return self.model(x)  
  
def compute_enhanced_depreciation_metrics(defects, image_shape=(256, 256), use_basic_features_only=True):  
    """計算增強版折舊指標，可選擇只使用基礎特徵"""  
    if not defects:  
        basic_metrics = {  
            "defect_index": 0, "defect_count": 0, "avg_depth": 0, "max_depth": 0, "total_area": 0  
        }  
        if use_basic_features_only:  
            return basic_metrics  
        else:  
            return {**basic_metrics, "depth_std": 0, "area_ratio": 0, "defect_density": 0}  
      
    # 基礎指標計算保持不變  
    defect_index = sum([d['area'] * d['depth'] for d in defects])  
    depths = [d['depth'] for d in defects]  
    areas = [d['area'] for d in defects]  
      
    avg_depth = np.mean(depths)  
    max_depth = np.max(depths)  
    total_area = sum(areas)  
      
    basic_metrics = {  
        "defect_index": defect_index,  
        "defect_count": len(defects),  
        "avg_depth": avg_depth,  
        "max_depth": max_depth,  
        "total_area": total_area  
    }  
      
    if use_basic_features_only:  
        return basic_metrics  
      
    # 額外特徵  
    depth_std = np.std(depths) if len(depths) > 1 else 0  
    total_image_area = image_shape[0] * image_shape[1]  
    area_ratio = total_area / total_image_area  
    defect_density = len(defects) / total_image_area * 10000  
      
    return {  
        **basic_metrics,  
        "depth_std": depth_std,  
        "area_ratio": area_ratio,  
        "defect_density": defect_density  
    }
  
def train_enhanced_mlp_from_csv(csv_path="depreciation_records.csv",   
                               output_path="enhanced_depreciation_mlp.pth",  
                               scaler_path="feature_scaler.pkl",  
                               epochs=100, patience=10):  
    """改良版 MLP 訓練函數  
    - 數據標準化  
    - 訓練/驗證分割  
    - 早停機制  
    - 學習率調度  
    - 類別權重平衡  
    """  
      
    # 讀取數據  
    df = pd.read_csv(csv_path)  
      
    # 特徵選擇（支援更多特徵）  
    feature_cols = ["defect_index", "avg_depth", "max_depth", "total_area"]  
    if "depth_std" in df.columns:  
        feature_cols.extend(["depth_std", "area_ratio", "defect_density"])  
      
    X = df[feature_cols].values  
    y = df["grade"].map({"A - normal": 0, "B - Under_observation": 1, "C - Recommended_repair": 2}).values  
      
    # 數據標準化  
    scaler = StandardScaler()  
    X_scaled = scaler.fit_transform(X)  
      
    # 儲存標準化器  
    with open(scaler_path, 'wb') as f:  
        pickle.dump(scaler, f)  
      
    # 訓練/驗證分割  
    X_train, X_val, y_train, y_val = train_test_split(  
        X_scaled, y, test_size=0.2, random_state=42, stratify=y  
    )  
      
    # 轉換為張量  
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)  
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)  
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)  
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)  
      
    # 計算類別權重  
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)  
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)  
      
    # 建立模型  
    model = EnhancedDepreciationMLP(input_dim=len(feature_cols))  
      
    # 載入已存在的模型（如果有）  
    if os.path.exists(output_path):  
        print(f"📂 載入已存在模型 {output_path}")  
        model.load_state_dict(torch.load(output_path, weights_only=True))  
      
    # 訓練設定  
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)  
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)  
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)  
      
    # 早停設定  
    best_val_loss = float('inf')  
    patience_counter = 0  
    best_model_state = None  
      
    # 訓練迴圈  
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)  
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)  
      
    model.train()  
    for epoch in range(epochs):  
        # 訓練階段  
        train_loss = 0  
        for batch_x, batch_y in train_loader:  
            optimizer.zero_grad()  
            logits = model(batch_x)  
            loss = criterion(logits, batch_y)  
            loss.backward()  
            optimizer.step()  
            train_loss += loss.item()  
          
        # 驗證階段  
        model.eval()  
        with torch.no_grad():  
            val_logits = model(X_val_tensor)  
            val_loss = criterion(val_logits, y_val_tensor).item()  
            val_acc = (torch.argmax(val_logits, dim=1) == y_val_tensor).float().mean().item()  
          
        # 學習率調度  
        scheduler.step(val_loss)  
          
        # 早停檢查  
        if val_loss < best_val_loss:  
            best_val_loss = val_loss  
            patience_counter = 0  
            best_model_state = model.state_dict().copy()  
        else:  
            patience_counter += 1  
          
        if epoch % 10 == 0:  
            print(f"Epoch {epoch}: Train Loss={train_loss/len(train_loader):.4f}, "  
                  f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")  
          
        if patience_counter >= patience:  
            print(f"早停於 epoch {epoch}")  
            break  
          
        model.train()  
      
    # 載入最佳模型  
    if best_model_state:  
        model.load_state_dict(best_model_state)  
      
    # 儲存模型  
    torch.save(model.state_dict(), output_path)  
    print(f"✅ 改良版模型已訓練並儲存至 {output_path}")  
      
    return model, scaler  
  
def classify_depreciation_enhanced_mlp(metrics, mlp_model, scaler):  
    """使用改良版 MLP 模型進行分類，自動適應特徵數量"""  
    # 根據 scaler 的特徵數量決定使用哪些特徵  
    scaler_features = scaler.n_features_in_  
      
    if scaler_features == 4:  
        # 只使用基礎特徵  
        features = [  
            metrics["defect_index"], metrics["avg_depth"],   
            metrics["max_depth"], metrics["total_area"]  
        ]  
    else:  
        # 使用所有特徵  
        features = [  
            metrics["defect_index"], metrics["avg_depth"],   
            metrics["max_depth"], metrics["total_area"],  
            metrics.get("depth_std", 0), metrics.get("area_ratio", 0),   
            metrics.get("defect_density", 0)  
        ]  
      
    # 其餘邏輯保持不變  
    features_scaled = scaler.transform([features])  
    input_tensor = torch.tensor(features_scaled, dtype=torch.float32)  
      
    with torch.no_grad():  
        logits = mlp_model(input_tensor)  
        probs = torch.softmax(logits, dim=1)  
        pred = torch.argmax(probs, dim=1).item()  
        confidence = probs[0, pred].item()  
          
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).item()  
        uncertainty = entropy / np.log(3)  
      
    labels = ["A - normal", "B - Under_observation", "C - Recommended_repair"]  
    return labels[pred], confidence, uncertainty 
  
def generate_enhanced_depreciation_record(defects, mlp_model=None, scaler=None, image_shape=(256, 256)):  
    """生成增強版折舊分析記錄，自動適應特徵數量"""  
    # 根據是否有 scaler 決定特徵類型  
    use_basic_only = scaler is None or scaler.n_features_in_ == 4  
    metrics = compute_enhanced_depreciation_metrics(defects, image_shape, use_basic_features_only=use_basic_only)  
      
    if mlp_model and scaler:  
        grade, confidence, uncertainty = classify_depreciation_enhanced_mlp(metrics, mlp_model, scaler)  
    else:  
        from train_depreciation_mlp import classify_depreciation  
        grade = classify_depreciation(metrics["defect_index"])  
        confidence = "N/A"  
        uncertainty = "N/A"  
      
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")  
    return {  
        "timestamp": timestamp,  
        "grade": grade,  
        "confidence": confidence,  
        "uncertainty": uncertainty,  
        **metrics,  
        "defects": defects  
    }