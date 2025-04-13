
# 🏗️ Gantry Crane Anti-Swing Control Simulation

🎯 **Anti-Swing Control using GA-Based LQR Controller for a Double Pendulum Gantry Crane with Load Hoisting and Lowering**

This repository contains the **academic research simulation** of a **double pendulum gantry crane system** equipped with **hoisting and lowering control**, utilizing a **Linear Quadratic Regulator (LQR)** optimized by **Genetic Algorithms (GA)**. The aim is to minimize payload swing and enhance control performance.

> 🚧 **This work is part of an ongoing research paper** and is currently **not released as open source**. Please refrain from unauthorized use.

---

## 📄 Abstract & Research Context

Developed for my graduate research paper:

> **"Anti-Swing Control using GA-Based LQR Controller for a Double Pendulum Gantry Crane with Load Hoisting and Lowering"**

This project explores robust anti-swing control of an underactuated gantry crane system. By leveraging **Genetic Algorithm optimization**, it adapts LQR gains for varying crane configurations, delivering improved swing suppression and energy-efficient motion during dynamic load handling.

📌 _This simulation is intended strictly for academic evaluation and peer review._

---

## 🎯 Objectives

- 🔄 Minimize swing of both pendulums under trajectory motion
- 🧠 Optimize LQR gains via GA for robust control
- 📉 Maintain smooth load hoisting and lowering
- 📊 Produce plots for performance analysis and comparison

---

## 🧰 Technologies

- `Python` for simulation scripting
- `NumPy`, `Pandas` for computation and data handling
- `Matplotlib` for visualization
- Custom implementations for:
  - Crane dynamics modeling
  - Genetic Algorithm
  - LQR formulation

---

## 📁 Project Structure

```bash
GantryCraneSimulation/
├── config/                    # Simulation configuration files (YAML)
├── scripts/                   # Execution scripts for simulation and GA tuning
├── src/                       # Modular code base
│   ├── controllers/           # LQR controllers and tuning logic
│   ├── core/                  # Crane dynamics models
│   ├── optimization/          # Genetic Algorithm implementation
│   ├── simulation/            # Integration of models with control strategies
│   ├── utils/                 # Helper methods and LQR calculation utilities
│   └── visualization/         # Result plotting and figure generation
├── simulation_plots/          # Output figures: trajectory, swing, energy, etc.
├── tests/                     # Unit tests and demonstration simulations
├── Notes.txt                  # Experimentation notes
└── README.md                  # This file
```

---

## ▶️ How to Run

1. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2. Run the LQR-GA optimization:
    ```bash
    python scripts/run_optimization.py
    ```

3. Simulate system response:
    ```bash
    python scripts/run_simulation.py
    ```

4. View output plots in the `simulation_plots/` directory.

---

## 📈 Sample Outputs

The simulation generates the following:
- 📉 Angular swing suppression
- 🚀 Smooth trajectory tracking
- ⚖️ Energy-efficient control signals
- 📊 GA evolution curves

---

## 📢 Disclaimer

This repository is part of a **research paper** in progress and the code is intended for demonstration and documentation purposes only.

📎 _Please do not redistribute or reuse the code without explicit permission._

---

## 👨‍🔬 Author & Contact

**Hytham Tag**  
Assistant Lecturer – Mechatronics/Robotics Engineering  
Benha University  
📧 haitham.adel@bhit.bu.edu.eg

---

> 🧪 _Research-driven engineering for smarter motion control._
