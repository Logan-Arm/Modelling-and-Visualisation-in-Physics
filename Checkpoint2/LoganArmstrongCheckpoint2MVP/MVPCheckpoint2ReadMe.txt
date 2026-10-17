# Cellular Dynamics Description

Two lattice-based simulations implementing Conway's Game of Life and the SIRS epidemic model. Both are run from the command line and support animation, data collection, and CSV export.


## Game of Life (GameOfLife.py)

A 2D cellular periodic lattice. Each cell is either alive (1) or dead (0), and evolves each step according to the standard rules: a live cell survives with 2 or 3 live neighbours, a dead cell is born with exactly 3 live neighbours, and all other cells die or remain dead.

### Initial Conditions (--Seed)

 Seed  & Description
| 0 | Random initialisation |
| 1 | Single 2x2 block — absorbing (still life) state |
| 2 | Several blinkers distributed across the lattice — dynamic equilibrium (period-2 oscillators) |
| 3 | Medium spaceship starting on the left-hand side — used for glider velocity analysis |

### Command Line Arguments for Game Of Life

Name, Type,  Value, Description

| --Width | int | 50 | Width of the lattice |
| --Length | int | 50 | Length of the lattice |
| --Seed | int | 0 | Initial condition (see above) |
| --Anim | bool | False | Animate the lattice evolution |
| --Data | int | 0 | Data collection mode (see below) |
| --NumberSims | int | 1000 | Number of independent simulations for convergence analysis |
| --NumSweeps | int | 1500 | Number of steps per run |
| --SaveData | str | None | Filename (without extension) to save results as CSV |

### Data Collection Modes (--Data)

- 0 — No data collection, runs the lattice with or without animation
- 1 — Convergence analysis: runs many random initialisations and records how many steps each takes to reach a stable live count (defined as 10 consecutive steps with the same number of live cells). Plots a histogram of results. Configurations that do not converge within 2000 steps are excluded.
- 2 — Glider analysis: runs the medium spaceship and tracks its centre of mass over time, plotting both distance from the initial position and instantaneous velocity.

### Output

Running with --SaveData saves three columns to a CSV: convergence step counts, glider distance, and glider velocity.



## SIRS Model (SIRS.py)

A stochastic epidemic model on a 2D periodic lattice. Each cell is in one of three states: susceptible (0), infected (1), or recovered (-1). A fourth state, permanently immune (-2), is available via seed 3. At each step, a single cell is chosen at random and updated probabilistically based on its state and its four nearest neighbours.

The transition rules are:
- Susceptible to infected with probability P_SI, if at least one nearest neighbour is infected
- Infected to recovered with probability P_IR
- Recovered to susceptible with probability P_RS

### Initial Conditions (--Seed)

Seed & Description
| 0 | Random initialisation using command line probabilities |
| 1 | Dynamic equilibrium — fixed at P_SI=0.3, P_IR=0.4, P_RS=0.3 |
| 2 | Absorbing state — fixed at P_SI=0.15, P_IR=0.55, P_RS=0.3 (infection dies out) |
| 3 | Random initialisation with a fraction of cells set permanently immune, fraction controlled by --P_Immune |

### Command Line Arguments for SIRS


Name, Type,  Value, Description
| --Width | int | 50 | Width of the lattice |
| --Length | int | 50 | Length of the lattice |
| --P_SI | float | 0.5 | Probability of susceptible becoming infected |
| --P_IR | float | 0.5 | Probability of infected becoming recovered |
| --P_RS | float | 0.5 | Probability of recovered becoming susceptible |
| --P_Immune | float | 0 | Fraction of the population that is permanently immune (seed 3 only) |
| --Seed | int | 0 | Initial condition (see above) |
| --Anim | bool | False | Animate the lattice evolution |
| --Data | int | 0 | Data collection mode (see below) |
| --NumberSims | int | 1000 | Number of frames for wave simulation |
| --NumSweeps | int | 1500 | Number of sweeps per configuration |
| --Res | int | 20 | Resolution (number of steps) for parameter sweeps |
| --EquibSweeps | int | 100 | Number of equilibration sweeps before data collection |
| --Prob_iminit | float | 0 | Starting immunity fraction for immunity sweep |
| --Prob_imfinal | float | 1 | Ending immunity fraction for immunity sweep |
| --SaveData | str | None | Filename (without extension) to save results as CSV |

### Data Collection Modes (--Data)

- 0 — No data collection, runs the lattice with or without animation
- 1 — Immunity sweep: varies the fraction of immune individuals from --Prob_iminit to --Prob_imfinal and plots the average infected fraction at each level
- 2 — Heatmap: sweeps both P_SI and P_RS across [0,1] at a given resolution and plots the average infected fraction as a 2D heatmap
- 3 — Infection variance: sweeps P_SI across [0,1] at fixed P_IR=P_RS=0.5 and plots the variance in infected fraction with bootstrap error bars, useful for identifying the phase transition
- 4 — Wave simulation: runs a fixed configuration (P_SI=0.8, P_IR=0.1, P_RS=0.01) that produces spiral wave patterns, then plots the infected, susceptible, and recovered counts over time

### Notes on Equilibration & Discarded Sweeps

The first 500 frames of any animation and the first 100 sweeps of any data collection run are treated as equilibration and excluded from measurements. The infected fraction is recorded per sweep (after each N=length×width individual updates).

### Output

Running with --SaveData saves immunity fractions, average infected fractions, and variance error estimates to a CSV.