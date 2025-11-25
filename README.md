# MLOPs Course

<div align="center">
  <img src="https://img.shields.io/github/stars/techiescamp/mlops-course.svg?style=for-the-badge" alt="GitHub Stars" />
  <img src="https://img.shields.io/github/forks/techiescamp/mlops-course.svg?style=for-the-badge" alt="GitHub Forks" />
  <img src="https://img.shields.io/github/contributors/techiescamp/mlops-course.svg?style=for-the-badge" alt="Contributors" />
  <img src="https://img.shields.io/github/last-commit/techiescamp/mlops-course/main.svg?style=for-the-badge" alt="Last Commit" />
  <img src="https://img.shields.io/badge/python-3.12.x-blue?style=for-the-badge" alt="Python Version" />
</div>

<hr />


## Table of Contents

- [Overview](#overview)
- [Projects](#projects)
- [Datasets](#datasets)
- [Setup & Installation](#setup--installation)
    - [Clone repo](#step-1-clone-the-repository)
    - [Create Virtual Environment](#step-2-create-virtual-environment)
    - [Install Dependencies](#step-3-install-dependencies)
- [Contribution](#contribution)
- [References](#references)


## Overview

This repository contains a series of hands-on projects designed to help DevOps engineers and developers understand the fundamentals of Machine Learning Operations (MLOps). 

Each project focuses on a specific stage of the ML lifecycle—ranging from data preparation to model deployment—so you can build practical, job-ready skills.

Below is an overview of the upcoming projects included in this course.

## Projects

```bash

mlops-course
|__ animal-classifier/
|__ ..

```

### 1. ML Basics for DevOps Engineers

Learn the essential concepts behind machine learning—algorithms, models, inferencing, and deployment at local development from a DevOps perspective.

**Note:** A lightweight introduction to prepare you for the hands-on projects ahead.

**Project: [animal-classifier/](./animal-classifier/README.md)**


## Datasets
Most of the datasets are taken from [UCI Zoo Dataset](https://archive.ics.uci.edu/dataset/) and [Kaggle](https://www.kaggle.com/datasets/)


## Setup & Installation

#### Step-1: Clone the repository

```bash
git clone https://github.com/techiescamp/mlops-course.git
cd animal-classifier
```

#### Step-2: Create Virtual Environment

##### Windows

```bash
python -m venv venv
venv\Scripts\activate
```
##### Mac and Linux
```bash
python3 -m venv venv
source venv/bin/activate
```
#### Step-3: Install Dependencies

```bash
pip install -r requirements.txt
```

#### Step-4: Run the Model Workflow step-by-step

Execute the code according to the instructions given each of the project's `readme.md` file. 


## Contribution
Please read our [Contributing Guidelines](CONTRIBUTION.md) before submitting pull requests.

## License
This project is under [MIT Licence](LICENCE) support.
