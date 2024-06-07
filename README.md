# SUMO-Routing-RL

## Summary

SUMO-Routing-RL is a reinforcement learning environment using the SUMO (Simulation of Urban MObility) traffic simulator. This project aims to optimize vehicle routing for community vehicles based on passenger disabilities and the availability of public transportation options.

## Features

- Integration with SUMO for realistic traffic simulation
- Multiple reinforcement learning algorithms including DQN and PPO
- Modular design for easy extension and experimentation
- Support for prioritized experience replay

## Table of Contents

- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [SUMO Installation](#sumo-installation)
  - [Project Setup](#project-setup)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

## Installation

### Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.8 or higher
- Anaconda or Miniconda (recommended for managing dependencies)

### SUMO Installation

Follow the official [SUMO installation instructions](https://sumo.dlr.de/docs/Installing/index.html).

### Project Setup

1. **Clone this repository**:
   ```bash
   git clone https://github.com/yourusername/SUMO-Routing-RL.git
   cd SUMO-Routing-RL
   ```

2. **Create and activate a conda environment**:
   ```bash
   conda create --name sumorl python=3.8
   conda activate sumorl
   ```

3. **Install project dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install SUMO as a Python package**:
   ```bash
   pip install sumolib traci
   ```

## Usage

To run the main training loop:

```bash
python src/main.py
```

To run with specific configurations:

```bash
python src/main.py --config path/to/config.yaml
```

**Note for Windows users:** Please use the `connect_no_gui` function instead of `connect_libsumo_no_gui` as `libsumo` does not work well on Windows.

## Project Structure

```
SUMO-Routing-RL/
┣ data/
┃ ┣ 3x3/
┃ ┃ ┗ Nets/
┃ ┃   ┣ 3x3.net.xml
┃ ┃   ┣ 3x3.rou.xml
┃ ┃   ┣ 3x3.sumocfg
┃ ┃   ┗ parkingareas.add.xml
┃ ┣ balt/
┃ ┃ ┗ Nets/
┃ ┃   ┣ osm.net.xml
┃ ┃   ┣ osm.sumocfg
┃ ┃   ┗ osm_pt.rou.xml
┣ docs/
┃ ┣ source/
┃ ┃ ┣ _static/
┃ ┃ ┣ _templates/
┃ ┃ ┣ conf.py
┃ ┃ ┗ index.rst
┃ ┣ Makefile
┃ ┗ make.bat
┣ notebooks/
┣ scripts/
┣ src/
┃ ┣ config/
┃ ┃ ┗ config.yaml
┃ ┣ environment/
┃ ┃ ┣ bus_stop.py
┃ ┃ ┣ env.py
┃ ┃ ┣ observation.py
┃ ┃ ┣ outmask.py
┃ ┃ ┣ person.py
┃ ┃ ┣ person_manager.py
┃ ┃ ┣ reward.py
┃ ┃ ┣ ride_select.py
┃ ┃ ┣ sim_manager.py
┃ ┃ ┣ vehicle.py
┃ ┃ ┗ vehicle_manager.py
┃ ┣ rl_algorithms/
┃ ┃ ┣ dqn/
┃ ┃ ┃ ┣ agent.py
┃ ┃ ┃ ┗ peragent.py
┃ ┃ ┣ exploration/
┃ ┃ ┗ exploration.py
┃ ┃ ┣ memory_buffers/
┃ ┃ ┃ ┗ replay_memory.py
┃ ┃ ┣ models/
┃ ┃ ┃ ┗ dqn.py
┃ ┃ ┣ ppo/
┃ ┃ ┗ ppoagent.py
┃ ┣ utils/
┃ ┃ ┗ utils.py
┃ ┗ main.py
┣ tests/
┣ .gitignore
┣ LICENSE
┣ README.md
┗ requirements.txt
```

## Documentation

Generate the documentation using Sphinx:

1. **Navigate to the `docs` directory**:
   ```bash
   cd docs
   ```

2. **Build the documentation**:
   ```bash
   make html
   ```

3. **Open the generated documentation**:
   Open `docs/build/html/index.html` in your web browser.

## Contributing

To contribute to this project, please follow these steps:

1. Fork this repository.
2. Create a branch: `git checkout -b feature/your-feature`.
3. Make your changes and commit them: `git commit -m 'Add your feature'`.
4. Push to the original branch: `git push origin feature/your-feature`.
5. Create a pull request.

Alternatively, see the GitHub documentation on [creating a pull request](https://help.github.com/articles/creating-a-pull-request).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Citation

If you use this project in your research, please cite it as follows:

```
@misc{ahope2024sumo,
  author = {Your Name},
  title = {SUMO-Routing-RL: Reinforcement Learning Environment Using SUMO Traffic Simulator},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yourusername/SUMO-Routing-RL}},
}
```
