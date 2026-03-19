※日本語ページは[こちら](https://github.com/crowd4u/crowd4u-aggregation-toolkit/blob/main/README_ja.md)です。

# Crowd4U Aggregation Toolkit

This toolkit provides implementations of aggregation methods and usage instructions to improve the quality of crowdsourcing results for requesters using [Crowd4U](https://crowd4u.org/).

## What is Aggregation?

<img width="600" height="230" alt="image" src="https://github.com/user-attachments/assets/77513b60-0041-45d1-9f54-80313197dfa8" />

In the post-processing of multi-class classification crowdsourcing, it is common practice to assign the same task to multiple workers and aggregate their results to mitigate the impact of individual worker errors. 
The simplest aggregation method is Majority Voting (MV). However, since worker competence varies significantly in crowdsourcing—and low-quality workers may even form the majority—Majority Voting is not always the optimal approach.

<img width="600" height="200" alt="image" src="https://github.com/user-attachments/assets/e0328e92-2af8-43a4-ba90-346188bb9414" />

Consequently, the standard approach involves using unsupervised learning with latent class models to estimate worker ability and performing weighted majority voting based on those estimates.

This repository provides implementations of these unsupervised learning-based aggregation methods in addition to simple Majority Voting. Furthermore, it includes aggregation methods specifically designed for "AI Workers," a unique feature of Crowd4U.

## Usage

### 1. Set Up the Docker Environment
To simplify environment setup, this toolkit uses Docker, a virtualization container environment. 
First, please set up a Docker environment on your computer (Windows, Mac, or Linux).

Refer to the official Docker manuals for installation instructions:
- Windows: https://docs.docker.com/desktop/install/windows-install/
- Mac (Apple Silicon): https://docs.docker.com/desktop/install/mac-install/
- Linux: https://docs.docker.com/desktop/install/linux-install/

*Note: Operation on ARM architectures (e.g., Apple Silicon) has not been verified. Execution on x64 architecture is recommended.*

### 2. Download the Toolkit
#### If you have git installed:
Clone this repository.
#### If you do not have or know how to use git:
<img width="500" height="252" alt="image" src="https://github.com/user-attachments/assets/9a103df4-5e7d-49f2-9d3b-e4acab1b7d44" />

1. Visit: https://github.com/crowd4u/crowd4u-aggregation-toolkit/
2. Click the green "Code" button in the upper right (circled in red in the image above) and select "Download ZIP."
3. Extract the downloaded ZIP file and save it to your preferred location.

### 3. Format and Save Crowdsourcing Data
This toolkit only accepts CSV files in the following format as input data. 
If you are using AI workers, you must save the human worker results and AI worker results in separate CSV files.

```csv
task,worker,label
question1,worker1,dog
question1,worker2,cat
question1,worker3,dog
question2,worker1,dog
question2,worker2,hamster
question2,worker3,hamster
question3,worker1,parrot
question3,worker2,parrot
question3,worker3,cat
```

- `task`: Task ID
- `worker`: Worker ID
- `label`: The answer label

Additionally, you must save the list of possible label values in a separate JSON file:

```json
[
    "dog",
    "cat",
    "rabbit",
    "hamster",
    "parrot"
]
```

**These files must be saved under the `datasets/` directory of this toolkit.** Sample data is provided in `datasets/` for your reference.

**Note: This toolkit provides a notebook to assist in converting Crowd4U TAR table data into the toolkit's format. Please refer to the notebook below if you need to convert from TAR format. Alternatively, data conversion can be easily performed using Generative AI tools.**

- wip

### 4. Run the Aggregation
1. Open a terminal where the docker command is available.
2. Navigate to the toolkit's folder in the terminal.
3. Run the following command to start the Docker container (it may take some time during the first run):
```sh
docker compose up -d
```
4. Once started, execute the aggregation using the following command:
```sh
docker exec crowd4u-aggregation-toolkit python main.py <Method_Name> <Output_Filename> <Labels_JSON_Filename> <Human_Data_CSV_Filename> <AI_Data_CSV_Filename (Optional)>
```

- `<Method_Name>`: Specify the aggregation method. The following are available:
    - `MV`: Simple Majority Voting. Uses the [Crowd-Kit](https://github.com/Toloka/crowd-kit) implementation.
    - `DS`: Dawid-Skene method [(Dawid & Skene 1979)](https://doi.org/10.2307/2346806). Uses the Crowd-Kit implementation.
    - `GLAD`: GLAD method [(Whitehill et al. 2009)](https://dl.acm.org/doi/10.5555/2984093.2984321). Uses the Crowd-Kit implementation.
    - `BDS`: Bayesian Dawid-Skene method using Markov Chain Monte Carlo (MCMC) [(Paun et al. 2018)](https://aclanthology.org/Q18-1040/). The first run will take some time.
    - `HSDS_EM`: Human-Seeded Dawid-Skene (HS-DS) method (our paper; under review) implemented via the EM algorithm, designed to address the issue of unbalanced AI performance. This cannot be applied to human-only data. It may fail on very small datasets (like the sample data).
    - `HSDS_MCMC`: HS-DS method implemented via MCMC. This cannot be applied to human-only data. The first run will take some time.
- `<Output_Filename>`: Specify the filename for the aggregated results. Results are saved in the `results/` folder.
- `<Labels_JSON_Filename>`: The JSON file created in step 3.
- `<Human_Data_CSV_Filename>`: The human worker CSV file created in step 3.
- `<AI_Data_CSV_Filename (Optional)>`: The AI worker CSV file created in step 3. Not required if no AI data exists. For methods that do not distinguish between humans and AI (any method other than HS-DS), this data will be vertically concatenated with the human data for processing.

**Examples:**
```sh
docker exec crowd4u-aggregation-toolkit python main.py MV sample_result.csv sample_labels.json sample_human.csv 
```
```sh
docker exec crowd4u-aggregation-toolkit python main.py DS sample_result.csv sample_labels.json sample_human.csv sample_ai.csv
```
```sh
docker exec crowd4u-aggregation-toolkit python main.py HSDS_MCMC sample_result.csv sample_labels.json sample_human.csv sample_ai.csv
```

5. The results will be saved in `results/` under the specified filename.
6. To stop the container, run the following command:
```sh
docker compose stop
```

## Contributors
The following members contribute to the development of this toolkit:

- Takumi TAMURA: https://takumi1001.github.io/takumi1001/index_en.html

For inquiries regarding this toolkit or Crowd4U, please refer to the following contact information:

https://fusioncomplab.org/about.html
