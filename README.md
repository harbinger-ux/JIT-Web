Here is the updated `README.md` file, retaining your existing setup instructions and inserting the new model weight downloading step.

```markdown
# MedCLIP Image Inference WebApp

A web-based medical image classification application powered by MedCLIP. This project provides a Flask backend and a dynamic web interface to run zero-shot and fine-tuned inference on medical scans (Axial and Sagittal planes) using PyTorch. Public access is routed securely via Ngrok.[cite: 10]

## Prerequisites

Before you begin, ensure your server or local machine has the following installed:[cite: 10]
* **NVIDIA GPU** with CUDA support (The script defaults to `cuda:0` / physical GPU 1).[cite: 10]
* **Conda** (Miniconda or Anaconda) for environment management.[cite: 10]
* **Ngrok** installed for exposing the local server to the internet.[cite: 10]

## 1. Installation & Setup

**Step 1: Navigate to the project directory**[cite: 10]
```bash
cd ~/AI-Healthcare/JIT-Webapp

```

**Step 2: Create the Conda environment**


The project dependencies are strictly locked in the `environment.yml` file. This command will download and install all necessary packages (including PyTorch, Flask, and CUDA toolkits) into an environment named `medclip-q2`.

```bash
conda env create -f environment.yml

```

**Step 3: Activate the environment**

```bash
conda activate medclip-q2

```

**Step 4: Download Model Weights**
Due to GitHub's file size limits, the large `.bin` weight files are hosted externally. Run the provided shell script to automatically fetch them from Google Drive and place them in the correct directories.

```bash
chmod +x download_weights.sh
./download_weights.sh

```

**Step 5: Authenticate Ngrok (First-time only)**


If you haven't linked your Ngrok account to your server yet, add your authentication token. You can find this token in your Ngrok dashboard under **Getting Started > Your Authtoken**.

```bash
ngrok config add-authtoken YOUR_AUTHTOKEN_HERE

```

## 2. Directory Structure

Ensure your project maintains the following structure for absolute path resolution to work correctly:

```text
JIT-Webapp/
├── app.py                  # Flask backend server
├── inference.py            # Model loading and prediction logic
├── environment.yml         # Conda environment configuration
├── download_weights.sh     # Script to fetch weights from Google Drive
├── medclip/                # Core MedCLIP source code and modules
├── templates/              # Web frontend interface
│   └── index.html          
├── text_val/               # Text validation files and DRW weights
│   ├── Mendeley/           
│   └── SPIDER/             
└── weights/                # Saved PyTorch model checkpoints
    ├── Mendeley/           
    └── SPIDER/             

```

## 3. Running the Web Application

You can run the application either in standard mode (great for debugging) or detached background mode (great for permanent deployment).

### Option A: Standard Execution (Requires 2 Terminals)



**Terminal 1: Start the Flask Backend**

```bash
conda activate medclip-q2
cd ~/AI-Healthcare/JIT-Webapp
python app.py

```

*The Flask server will start locally on `http://0.0.0.0:5001`.*

**Terminal 2: Start the Ngrok Tunnel**

```bash
ngrok http 5001

```

*Ngrok will display a terminal UI. Copy the HTTPS **Forwarding** URL and open it in your browser.*

---

### Option B: Permanent Background Execution (Recommended)



If you want the web app to stay online even after you close your SSH session, use the `nohup` (no hangup) command.

**Start the processes in the background:**

```bash
conda activate medclip-q2
cd ~/AI-Healthcare/JIT-Webapp

# Start the Flask app
nohup python app.py > webapp.log 2>&1 &

# Start the Ngrok tunnel
nohup ngrok http 5001 > ngrok.log 2>&1 &

```

**Retrieve your public Ngrok URL:**


Because Ngrok is running silently, use this command to query its local API and print your public access link:

```bash
curl -s [http://127.0.0.1:4040/api/tunnels](http://127.0.0.1:4040/api/tunnels) | python3 -c "import sys, json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])"

```

## 4. Usage

1. Open the provided Ngrok URL in any web browser.


2. Upload a medical image scan (PNG, JPG, JPEG).


3. Select the anatomical plane corresponding to the image:


* **Axial** (Routes to the Mendeley dataset weights)


* **Sagittal** (Routes to the SPIDER dataset weights)




4. Adjust the **Positivity Threshold** (0.0 to 1.0) if you wish to change the sensitivity of the binary prediction labels.


5. Click **Run Inference**.



## 5. Troubleshooting & Maintenance

* **Check Flask Logs:** If the background app crashes or you want to monitor inference times, read the log file:


```bash
tail -f webapp.log

```


* **Check Ngrok Logs:**

```bash
tail -f ngrok.log

```


* **Stop Background Processes:** To completely shut down the permanent server, find the process IDs and kill them:


```bash
pkill -f "python app.py"
pkill -f "ngrok http 5001"

```



```

```