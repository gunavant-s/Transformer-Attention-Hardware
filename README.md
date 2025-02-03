# VLSI Implementation of Transformer Scaled Dot-Product Attention Module for NLP


## Project Overview
This project implements a hardware accelerator for the **Scaled Dot-Product Attention** mechanism, a core component of transformer models widely used in Natural Language Processing (NLP) and Artificial Intelligence (AI). The design focuses on efficient computation of matrix multiplications for Query (Q), Key (K), and Value (V) matrices, followed by attention score calculation and final output generation. The hardware is implemented in Verilog/SystemVerilog and optimized for performance, area, and power efficiency.

---

## Features
- **Efficient Matrix Multiplication**: Implements parallel processing for Q, K, and V matrices.
- **SRAM Management**: Optimized use of input, weight, result, and scratchpad SRAMs for intermediate data storage.
- **Finite State Machine (FSM)**: A robust FSM controls the computation flow across 21 states.
- **Dynamic Address Calculation**: Dynamically calculates SRAM read/write addresses for efficient memory access.
- **Scalability**: Designed to handle larger matrices with minimal modifications.

---

## Directory Structure
```
├── src/
│   └── dut.v               # Top-level module integrating datapath and control
├── testbench/
│   └── tb_fsm.v            # Testbench for FSM
|   └── sram.v              # SRAM read and write
```

---

## How It Works

### 1. FSM Control
The design uses a 21-state FSM to manage the computation flow:
- **States S0-S5**: Compute Q matrix.
- **States S6-S8**: Compute K matrix.
- **States S9-S11**: Compute V matrix.
- **States S12-S19**: Compute final attention scores (S) and output matrix (Z).
- **State S20**: Reset and prepare for the next input.

### 2. Datapath
The datapath performs:
- Multiply-accumulate (MAC) operations to compute \( Q \times K^T \), scale by \( \sqrt{d_k} \), and apply softmax.
- Accumulation of results into SRAM for intermediate storage.
- Dynamic address calculation based on counters and control signals.

### 3. Memory Management
The design uses multiple SRAMs:
- **Input SRAM**: Stores input embeddings.
- **Weight SRAM**: Stores learned weights (\( W_Q \), \( W_K \), \( W_V \)).
- **Result SRAM**: Stores intermediate results like \( QK^T \), attention scores, and final outputs.
- **Scratchpad SRAM**: Temporary storage during computations.

---

## How to Run

### Prerequisites
Ensure you have the following tools installed:
1. Verilog/SystemVerilog simulator (e.g., ModelSim or VCS)
2. Synthesis tool (e.g., Synopsys Design Compiler)
3. GTKWave (for waveform viewing)

### Simulation Steps
1. Clone this repository:
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```
2. Compile the design files:
   ```bash
   make compile
   ```
3. Run the testbench:
   ```bash
   make simulate
   ```
4. View waveforms using GTKWave:
   ```bash
   gtkwave waveforms.vcd
   ```

### Synthesis Steps
1. Run synthesis using Synopsys Design Compiler:
   ```bash
   make synthesize
   ```
2. Check synthesis reports in the `synthesis/` directory.

---

## Results

### Key Metrics:
| Metric               | Value                     |
|----------------------|---------------------------|
| Total States         | 21                        |
| Area                 | ~15,257 units            |
| Worst Negative Slack | 3.37 ns                   |
| Leakage Power        | ~222.82 million units     |
| Throughput           | 3,384 cycles per attention computation |

### Highlights:
1. Efficient parallel processing reduces total execution time.
2. Optimized memory management minimizes latency during computations.
3. FSM ensures precise control over computation flow.

---

## Files Description

| File Name             | Description                                                                   |
|-----------------------|-------------------------------------------------------------------------------|
| `dut.v`               | The logic for this project consisting of datapath, control path and interfaces|
| `sram.v`              | Handles SRAM read/write operations efficiently with dynamic addressing.       |
| `testbench.v`         | Testbench for verifying the functionality of the top module.                  |

---

## Future Work

1. Optimize dynamic power consumption through clock gating or reduced switching activity.
2. Integrate DesignWare IP for square root computation to improve efficiency in \( \frac{QK^T}{\sqrt{d_k}} \).
3. Scale the design to support larger transformer models by increasing matrix dimensions.

---

## Contributors

This project was developed as part of an academic hardware design course ECE 564 ASIC AND FPGA DESIGN USING VERILOG

---
