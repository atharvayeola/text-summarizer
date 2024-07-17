
# Text Summarizer Project

## Overview
This project is an end-to-end text summarizer application. It uses NLP techniques to provide concise and meaningful summaries of long texts.

## Workflows
1. Update `config.yaml`
2. Update `params.yaml`
3. Update entity
4. Update the configuration manager in `src/config`
5. Update the components
6. Update the pipeline
7. Update `main.py`
8. Update `app.py`

## How to Run

### Steps:

1. **Clone the repository**
    ```bash
    git clone https://github.com/atharvayeola/text-summarizer
    ```

2. **Create a conda environment after opening the repository**
    ```bash
    conda create -n summary python=3.8 -y
    ```

    ```bash
    conda activate summary
    ```

3. **Install the requirements**
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the application**
    ```bash
    python app.py
    ```

5. **Open your local host and port to access the application**

## AWS CI/CD Deployment with GitHub Actions

1. **Login to AWS console.**

2. **Create IAM user for deployment with specific access:**
    - EC2 access: Virtual machine
    - ECR: Elastic Container Registry to save your Docker image in AWS

    **Description:**
    1. Build Docker image of the source code
    2. Push your Docker image to ECR
    3. Launch your EC2
    4. Pull your image from ECR in EC2
    5. Launch your Docker image in EC2

    **Policy:**
    - AmazonEC2ContainerRegistryFullAccess
    - AmazonEC2FullAccess

3. **Create ECR repository to store/save Docker image**
    - Save the URI: `<your ECR URI>`

4. **Create EC2 machine (Ubuntu)**

5. **Open EC2 and install Docker in EC2 machine:**
    ```bash
    sudo apt-get update -y
    sudo apt-get upgrade
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker ubuntu
    newgrp docker
    ```

6. **Configure EC2 as self-hosted runner:**
    Go to GitHub `Settings > Actions > Runners > New self-hosted runner`, choose OS, and then run commands one by one.

7. **Setup GitHub secrets:**
    - `AWS_ACCESS_KEY_ID`
    - `AWS_SECRET_ACCESS_KEY`
    - `AWS_REGION`: `us-east-1`
    - `AWS_ECR_LOGIN_URI`: `demo >> <your ECR login URI>`
    - `ECR_REPOSITORY_NAME`: `text-summarizer`

## Author
Atharva Yeola

Email: atharvayeola12@gmail.com
