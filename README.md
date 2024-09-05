# 🚀 AI Deep Q-Learning for Lunar Landing & OpenAI's Gymnasium

This repository contains the implementation of a **Deep Q-Learning (DQN) agent** to solve the Lunar Lander environment from OpenAI Gym. The objective is to train an agent to safely land a spacecraft on the moon’s surface using reinforcement learning techniques.

---

## 📚 Table of Contents
- [Introduction](#introduction)
- [Getting Started](#getting-started)
- [Requirements](#requirements)
- [Usage](#usage)
- [Results](#results)
- [Demo](#demo)
---

## 🌟 Introduction

The **Lunar Lander** environment is a popular reinforcement learning problem provided by OpenAI Gym. The goal is to safely land a spacecraft on the moon's surface using limited control of thrusters. In this project, we implement **Deep Q-Learning (DQN)** to train an agent that learns the optimal policy for successful landings.

Key objectives:
- Use a neural network to approximate Q-values.
- Train an agent to take actions based on the current state of the lunar module.
- Maximize rewards by safely landing the spacecraft with minimal fuel consumption and soft landings.

---

## 🚀 Getting Started

To get started with this project, follow the steps below:

1. Clone this repository:
bash
git clone https://github.com/melikaknight/Deep-Q-Learning-for-Lunar-Landing.git
cd Lunar-Lander-AI
2. Install the required dependencies:
It's recommended to use a virtual environment to manage dependencies:

bash
Copy code
pip3 install -r requirements.txt
---

## 🔧 Requirements
Ensure you have the following installed:

- **Python 3.7+**
- **OpenAI Gym**: Provides the Lunar Lander environment.
- **NumPy**: For efficient mathematical operations.
- **TensorFlow** or **PyTorch**: For building the neural network (based on your implementation).
- Other required libraries (see `requirements.txt`).

To install dependencies, run:
bash
pip install -r requirements.txt
---
## 💻Usage
To train the agent and observe its performance, follow these steps:

Open the Jupyter notebook:

bash
Copy code
jupyter notebook Deep Q-Learning for Lunar Landing.ipynb
Run all cells to:
Set up the environment.
Train the Deep Q-Learning agent.
Monitor the training progress and evaluate the agent's performance.

## 📊 Results
The trained agent learns to land the spacecraft successfully after several episodes of training. The performance can be measured using the following metrics:

- **Cumulative reward per episode**: Indicates how well the agent is performing over time.
- **Landing success rate**: Tracks the agent's ability to safely land.

### Key Outcomes:
- Successful landings with minimal crashes.
- Optimal fuel usage for efficient landings.

## 🎥 Demo
Watch the trained agent in action as it performs successful lunar landings!  
[![Demo CountPages alpha](https://github.com/melikaknight/Deep-Q-Learning-for-Lunar-Landing/blob/main/Video/Demo.gif)]
