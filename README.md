
# 🏗️ Gantry Crane Anti-Swing Control Simulation

🎯 **Anti-Swing Control using GA-Based LQR Controller for a Double Pendulum Gantry Crane with Load Hoisting and Lowering**

This repository hosts the simulation of a **double pendulum gantry crane system** that incorporates **hoisting/lowering control** and implements an **LQR controller** optimized via **Genetic Algorithms (GA)**. The system is modeled and tested in simulation with the objective of minimizing payload swing during dynamic movements.

> ⚠️ This is a **simulation-only project** and has not yet been implemented physically.

---

## 📄 Abstract & Research Context

This project was developed as part of my academic research paper titled:

> **"Anti-Swing Control using GA-Based LQR Controller for a Double Pendulum Gantry Crane with Load Hoisting and Lowering"**

The paper presents a creative approach that utilizes a **GA-tuned LQR controller** to handle the nonlinear dynamics of a gantry crane system under various mass and length configurations. The focus is on enhancing control robustness while reducing angular swing and trajectory error.

📌 _If you use this work in research, please cite the title above or contact me for citation details._

---

## 🧠 Project Goals

- ✅ Suppress swing angles in both pendulums during motion
- ✅ Implement optimal control using LQR + Genetic Algorithm tuning
- ✅ Handle dynamic load hoisting and lowering
- ✅ Visualize and evaluate control performance via simulation plots

---

## 🛠️ Technologies Used

- **Python**
- **LQR (Linear Quadratic Regulator)**
- **Genetic Algorithm Optimization**
- **Custom Crane Dynamics Model**
- Plotting & evaluation using **Matplotlib**, **NumPy**, **Pandas**

---

## 📁 Project Structure

```bash
GantryCraneProject
├── config/                    # YAML config files for simulations
├── scripts/                   # Optimization & simulation entry points
├── src/                       # Main source code (controllers, models, utils)
│   ├── controllers/           # LQR & adaptive control methods
│   ├── core/                  # Crane physics and dynamics
│   ├── optimization/          # GA, PSO, and optimizer base classes
│   ├── simulation/            # Crane simulation logic
│   ├── utils/                 # LQR handler, helper functions
│   └── visualization/         # Plot generation and exports
├── simulation_plots/          # Output plots for trajectory, energy, jerk, etc.
├── tests/                     # Testing scripts and trajectory demos
├── Notes.txt                  # Design notes
└── README.md                  # This file
```

---

## 🚀 How to Run

1. Clone the repo and install dependencies:
    ```bash
    git clone https://github.com/HythamTag/gantry-crane-lqr-ga.git
    cd gantry-crane-lqr-ga
    pip install -r requirements.txt
    ```

2. Run an optimization:
    ```bash
    python scripts/run_optimization.py
    ```

3. Simulate a trajectory:
    ```bash
    python scripts/run_simulation.py
    ```

4. View plots in the `simulation_plots/` folder.

---

## 📊 Sample Output

Simulation generates plots including:
- 🌀 Swing angle reduction
- 🧲 Control inputs
- 📈 Energy consumption
- 🛤️ Payload trajectory

---

## 🧹 Refactoring Notes

This codebase is functional but would benefit from:
- Modular configuration files
- Improved variable naming and cleanup
- Refactoring for better unit testing and real-time plotting support

---

## 📬 Contact

For questions, suggestions, or collaboration, feel free to reach out to:

**Hytham Tag**  
📧 [haitham.adel@bhit.bu.edu.eg]  

---

> Made with 💡 by Hytham Tag — Mechatronics Engineer & Robotics Researcher  
> 🧪 "Turning control theory into motion precision"
