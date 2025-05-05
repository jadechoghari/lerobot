# 🚀 Train Your Own Robot Policy with Hugging Face + LeRobot

In this guide, we’ll walk you through how to train your own robot policy using [LeRobot](https://huggingface.co/lerobot) and Hugging Face Spaces (with dev mode!). You’ll use your own task dataset, choose a policy, and train it—all inside a GPU-powered dev environment. Let's get started 💪

---

## 🧠 What's a Policy?

A **policy** is the "brain" of your robot. It takes in observations—like the robot’s joint positions, environment states, and even camera images—and outputs actions (like joint angles) via a neural network.

---

## 1. 🧪 Choose Your Task and Policy

First, pick a task! For this tutorial, we’ll use the `act_aloha_sim_insertion_human` dataset.
👉 **\[TODO: briefly describe the task]**

Then, select a policy. LeRobot supports several plug-and-play options—including some that even accept language as input (like `pi0` 🤯). In our case, we’ll use the **ACT** policy.

---

## 2. 🛠️ Set Up Your Space

Head over to [Hugging Face Spaces](https://huggingface.co/spaces) and click **“Create Space.”**

* Select **`GPU`** and pick something like a `T4` (good enough for this demo)
* Most importantly, choose **“Dev Mode”** to get a full terminal + SSH access!

📸 \[TODO: Add image of the setup screen]

---

## 3. 🔐 Log In & Clone LeRobot

Once your space is ready, click **“Open in Terminal”** or SSH in.

Clone the LeRobot repo and install it:

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e .
```

📸 \[TODO: Add screenshot of install]

---

## 4. 📦 Optional: Use `tmux`

We recommend using `tmux` so you don’t lose your training session if the terminal disconnects:

```bash
apt-get install tmux
tmux
```

---

## 5. 🏋️‍♂️ Start Training

Now run your training script!
👉 \[TODO: Add example training script]

Your weights will be saved in a folder like `./checkpoints/last`.

---

## 6. 🚢 Push to Hugging Face Hub

Once training is done:

1. Go to your [Hugging Face profile](https://huggingface.co/) → click **New Repo**
2. Clone your new model repo:

```bash
git lfs install
git clone https://huggingface.co/YOUR_USERNAME/YOUR_MODEL
cd YOUR_MODEL
```

3. Copy your final weights into this folder and push:

```bash
cp -r /path/to/checkpoints/last/* ./
git add .
git commit -m "Add trained policy weights"
git push
```

And that’s it! 🥳 Your policy is now live on the Hub and ready to be used/tested by anyone.
