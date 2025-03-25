# Model-Free Non-Stationary RL

This folder contains the code for the paper "Model-Free Non-Stationary RL: Near-Optimal Regret and Applications in Multi-Agent RL and Inventory Control". 


## Folder Structure


    .
    ├── README.md                   
    ├── Combination_Lock.py                 # Code for the Combination Lock environment
    ├── Double_RestartQ_UCB.py              # Implementation of Double-Restart Q-UCB algorithm
    ├── Epsilon_Greedy.py                   # Implementation of epsilon-greedy algorithm
    ├── LSVI_UCB_Restart.py                 # Implementation of LSVI-UCB-Restart algorithm
    ├── main.py                             # Main entrance of the project
    ├── Q_Learning_UCB.py                   # Implementation of Q-learning UCB algorithm
    └── RestartQ_UCB.py                     # Implementation of RestartQ-UCB algorithm


## Run

To run the code for any algorithm, simply specify the corresponding ```algo_id``` in ```main.py``` and run:

```
python main.py
```
