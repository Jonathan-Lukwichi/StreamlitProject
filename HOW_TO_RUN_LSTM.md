# How to Run LSTM (CRITICAL INSTRUCTIONS)

## ⚠️ IMPORTANT: You MUST Use the New Environment

Your current setup has **TWO** Python installations:
1. **Python 3.14** (your default) - ❌ **NO TensorFlow** - WILL FAIL
2. **Python 3.11** (forecast_env_py311/) - ✅ **HAS TensorFlow** - WORKS

## The Problem

When you run Streamlit normally, it uses Python 3.14 which doesn't have TensorFlow, so LSTM training fails with:
```
❌ All horizons failed during training
```

## The Solution

You MUST run Streamlit using the Python 3.11 environment.

## 🚀 HOW TO RUN (3 Methods)

### Method 1: Use the Batch File (EASIEST)

1. **CLOSE** any running Streamlit windows/terminals
2. Double-click: **`start_lstm.bat`**
3. You should see:
   ```
   [1/3] Activating Python 3.11 environment...
   [2/3] Verifying environment...
   Python 3.11.1
   TensorFlow: 2.15.1
   XGBoost: 3.1.1
   [3/3] Starting Streamlit...
   ```
4. Wait for Streamlit to open in your browser
5. Go to **Modeling Hub** (page 8)
6. Select **LSTM** and train!

### Method 2: Manual Command Line

1. Open **Command Prompt** (cmd.exe) in this folder
2. Run these commands **one by one**:
   ```cmd
   forecast_env_py311\Scripts\activate
   python --version
   ```
   ⚠️ You MUST see: `Python 3.11.1` (not 3.14!)

3. If you see 3.11.1, then run:
   ```cmd
   streamlit run pages\01_Dashboard.py
   ```

### Method 3: Direct Python Call (No Activation)

```cmd
forecast_env_py311\Scripts\python.exe -m streamlit run pages\01_Dashboard.py
```

## ✅ How to Verify You're Using the Right Environment

When Streamlit starts, you should see in the terminal:
```
Python 3.11.1
TensorFlow: 2.15.1
XGBoost: 3.1.1
```

If you see `Python 3.14.0` - **STOP** - you're using the wrong environment!

## 🔍 Troubleshooting

### Still Getting "All horizons failed"?

Check your terminal/console output. You should see:
```
🔍 LSTM PREPROCESSING DETECTION REPORT
======================================================================
Overall State: ...
Recommendation: ...
```

If you see errors about "No module named 'tensorflow'", you're **NOT** using the new environment.

### How to Stop Wrong Streamlit

If Streamlit is already running with Python 3.14:
1. Press `Ctrl+C` in the terminal
2. Close the terminal completely
3. Use one of the methods above to start with Python 3.11

## 📊 Testing LSTM

Once Streamlit is running with the correct environment:

1. Go to **Modeling Hub** (page 08_Modeling_Hub.py)
2. Select model type: **LSTM**
3. Configure (or use defaults):
   - Lookback Window: 14
   - LSTM Hidden Units: 64
   - Number of Layers: 2
   - Dropout: 0.2
   - Epochs: 100
   - Learning Rate: 0.001
   - Batch Size: 32
4. Click **Train Model**
5. You should see the preprocessing detection report in the console
6. Training will take 1-3 minutes depending on your hardware

## 🎯 Expected Output (Console)

```
🔍 LSTM PREPROCESSING DETECTION REPORT
======================================================================
Overall State: FULLY_SCALED / UNSCALED / PARTIALLY_SCALED
Recommendation: [specific recommendation]
Scaled Features: X/Y
----------------------------------------------------------------------
⚙️  APPLYING SCALING for consistency...
✓ Scaling applied
======================================================================
📊 Creating sequences (lookback=14)...
   Train sequences: (N, 14, features)
   Val sequences: (M, 14, features)
🏗️  Building LSTM model...
   Architecture: 2 layers, 64 units
🚀 Training LSTM (100 epochs max)...
✓ Training complete (X epochs)
======================================================================
```

## ❌ What NOT to Do

1. **DON'T** run `python pages/01_Dashboard.py` directly
2. **DON'T** run `streamlit run ...` without activating the environment first
3. **DON'T** use your global Python 3.14 installation

## ✅ What TO Do

1. **USE** `start_lstm.bat` (double-click it)
2. **OR** activate the environment first, then run Streamlit
3. **VERIFY** you see Python 3.11.1 when it starts

---

**Bottom Line**: Always use `start_lstm.bat` or manually activate `forecast_env_py311` before running Streamlit!
