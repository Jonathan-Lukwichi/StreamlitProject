# How to Run Streamlit with LSTM in VS Code

## ✅ Setup Complete!

I've configured VS Code to run Streamlit with Python 3.11 (TensorFlow support) using the Run/Debug button.

## 🚀 How to Run

### Method 1: Using the Run Button (EASIEST)

1. **Click** the "Run and Debug" icon in the left sidebar (or press `Ctrl+Shift+D`)
2. At the top, you'll see a dropdown menu
3. **Select**: `🚀 Streamlit with LSTM (Python 3.11 + TensorFlow)`
4. **Click** the green play button (▶) or press `F5`
5. Streamlit will start in the integrated terminal
6. Your browser should open automatically

![VS Code Run Button](https://i.imgur.com/placeholder.png)

### Method 2: Quick Launch

1. Press `F5` (if the correct configuration is already selected)
2. Streamlit will start immediately

### Method 3: From the Menu

1. Go to: `Run` → `Start Debugging` (or `Run Without Debugging`)
2. Select: `🚀 Streamlit with LSTM (Python 3.11 + TensorFlow)`

## 🧪 Testing the Environment

Before running Streamlit, you can test if TensorFlow is installed:

1. In the Run dropdown, select: `🧪 Test LSTM Environment`
2. Click the green play button (▶)
3. You should see ">>> ALL TESTS PASSED <<<" in the terminal

## 📋 Available Configurations

I've created 4 run configurations for you:

### 1. 🚀 Streamlit with LSTM (Python 3.11 + TensorFlow)
- **Use this for LSTM training!**
- Runs Streamlit with Python 3.11 and TensorFlow
- Starts the Dashboard page
- Opens in your browser automatically

### 2. 🧪 Test LSTM Environment
- Verifies TensorFlow and XGBoost are installed
- Tests LSTM and XGBoost pipeline imports
- Shows environment details

### 3. 🐍 Run Current File (Python 3.11)
- Runs whatever Python file you currently have open
- Uses Python 3.11 environment

### 4. Streamlit: run app (OLD - Python 3.14)
- **Don't use this!** It uses Python 3.14 without TensorFlow
- Kept for reference only

## ✅ Verify It's Working

When Streamlit starts, check the terminal output. You should see:
- Python 3.11 mentioned (not 3.14)
- No errors about missing TensorFlow
- When training LSTM, you'll see: `🔍 LSTM PREPROCESSING DETECTION REPORT`

## 🔧 Python Interpreter

VS Code is now configured to use Python 3.11 by default:
- **Interpreter**: `forecast_env_py311/Scripts/python.exe`
- You should see `Python 3.11.1` in the bottom-left status bar

If you don't see this:
1. Click the Python version in the bottom-left
2. Select: `forecast_env_py311/Scripts/python.exe`

## 🎯 Quick Test

1. Open VS Code's integrated terminal (`` Ctrl+` ``)
2. You should automatically be in the Python 3.11 environment
3. Run: `python --version`
4. Should show: `Python 3.11.1`

## ❌ Troubleshooting

### "Module 'streamlit' not found"
The Python extension might not have loaded the environment yet:
1. Close and reopen VS Code
2. Or manually select the interpreter (bottom-left status bar)

### LSTM still fails with "No module named 'tensorflow'"
Check which Python is being used:
1. Look at the terminal when Streamlit starts
2. Should see Python 3.11.1, not 3.14.0
3. If it's 3.14, the launch configuration didn't work
4. Try restarting VS Code

### Terminal shows Python 3.14
The launch configuration should override this:
1. Make sure you selected the correct configuration in the dropdown
2. The one with 🚀 and "(Python 3.11 + TensorFlow)"

## 📝 Next Steps

1. Press `F5` to launch Streamlit
2. Navigate to **Modeling Hub** (page 8)
3. Select **LSTM** model
4. Click **Train Model**
5. Watch the terminal for preprocessing detection report!

---

**Pro Tip**: You can also create keyboard shortcuts:
- `Ctrl+Shift+B` for quick Streamlit launch
- Configure in: File → Preferences → Keyboard Shortcuts → Search "launch"
