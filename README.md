<img src="DT-guided DRL.png" align="right" width="40%"/>

# Decision Theory-Guided Deep Reinforcement Learning for Fast Learning

Welcome to the official GitHub repository for the research paper "Decision Theory-Guided Deep Reinforcement Learning for Fast Learning". This project is at the forefront of combining Decision Theory with Deep Reinforcement Learning (DRL) to overcome the cold start problem, a notorious challenge in the realm of DRL applications.

## Research Abstract

This paper introduces a novel approach, Decision Theory-guided Deep Reinforcement Learning (**DT-guided DRL**), to address the inherent cold start problem in DRL. By integrating decision theory principles, DT-guided DRL enhances agents' initial performance and robustness in complex environments, enabling more efficient and reliable convergence during learning. Our investigation encompasses two primary problem contexts: the cart pole and maze navigation challenges. Experimental results demonstrate that the integration of decision theory not only facilitates effective initial guidance for DRL agents but also promotes a more structured and informed exploration strategy, particularly in environments characterized by large and intricate state spaces. The results of experiment demonstrate that DT-guided DRL can provide significantly higher rewards compared to regular DRL. Specifically, during the initial phase of training, the DT-guided DRL yields up to an 184% increase in accumulated reward. Moreover, even after reaching convergence, it maintains a superior performance, ending with up to 53% more reward than standard DRL in large maze problems. DT-guided DRL represents an advancement in mitigating a fundamental challenge of DRL by leveraging functions informed by human (designer) knowledge, setting a foundation for further research in this promising interdisciplinary domain.

### Highlights of Findings:

- **Cart Pole Challenge**: DT-guided DRL demonstrated superior initial rewards and expedited convergence towards optimal policies compared to conventional DRL strategies, highlighting the efficacy of decision theory-based heuristics in early learning stages.
- **Maze Navigation**: In the maze problem, DT-guided DRL consistently outperformed existing approaches across various maze sizes, showcasing its adaptability and effectiveness in complex environments with sparse rewards.
- **Structured Exploration**: The integration of decision theory not only provided effective initial guidance for the DRL agents but also contributed to a more structured and informed exploration strategy, particularly in environments with large state spaces and intricate navigational challenges.

## Repository Notes

- **`main.py`**: The central script for executing simulations. Adjust the simulation parameters within the `if __name__ == '__main__':` section to tailor the experiments to your requirements.
- **`figure_generate.py`**: A script designed to generate illustrative figures from Tensorboard log files, aiding in the clear presentation and analysis of research outcomes.
- **Stable Baselines3 Integration**: The DRL components of this project are built upon the framework provided by 'Stable Baselines3' (URL: [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io)).
- **Gym-Maze Customization**: The maze challenges are adapted from the versatile 'gym-maze' environment (URL: [Gym-Maze on GitHub](https://github.com/MattChanTK/gym-maze)).

This repository invites researchers and enthusiasts to delve into DT-guided DRL. Whether you're aiming to replicate our findings or innovate further, this codebase provides a foundation for exploring the integration between Decision Theory and Deep Reinforcement Learning.

