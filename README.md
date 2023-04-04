# BalitaNet - Filipino News Article Generator

This was made to explore the capabilities of current NLP architectures for language generation in the low-resource Filipino language.

Check out all the details in the [research paper](https://ffd6057631f7079ae73072492b526b4cf7ddbac3100a71a2487e6f9-apidata.googleusercontent.com/download/storage/v1/b/public-kenricklancebunag/o/Transformer-based%20Conditional%20Language%20Models%20-%20IEOM%20Submission.pdf?jk=Ac_6HjJwZmSk_Quyxf7rKKQwLJiNz7216Xs4a9WxJ2w13rwrLHOZw_2Q2JyrwGEAvj3myYUIeVTWJwdbpb2BBHwTbVlDhTb073-6v8FfwcByNjqok6ShiZxkAFQW7NPZfPmtCluJQ7kJvYOE_th1dEfMR5IQAdJQ-n6jHmKNA9mPEVUsJU5ofoWpubL1ARh499uIEZw7oyuZZhTJm58F8f0PdJGdQbGRgVwbVIydUUXjST2hh2pbsWFx-C3S65l4HILOzxvRSh8UWFCfsATTu0UmLXQJQZelXnmb6U3e63UdxhTt914yjqGH7gSAfoKJsIdskW6LKPXuWjgQPVhkcdUO6KtHCWdSP1jwj_y4Jvd7nP7LNyLAeakZ-raVz75kBGZQgtm9b5Yv9MR2NVfBTe5Tq03VKg7rE5btvWBsSfaTMj7SqaCPQSDfikPcdTitA556alAsykwPQ6hSHy-GomMiNcyxVFB46x7xlJldi2QNFn9NCcCEYYiIbig8Qea5ibdALzp5AIA6tvWMJkhdEi82MYuD9ei29XaW5HdD9vCcw8OraPs8TzuG82nDRV0sakVHV2M2lN9tlFgGNekqF4aYp30UrkIrf1NiwVJ-fP2rHJap8PLLATUv6FqBKeXUFAYZCEJM5LKd2yrg-Ew4u4hCdlAnGQly2dSHuSm1eMYu74g57dAreQYjCc9ZNA2v1EQPOk4USgMHOriNNHvtDcsD9f_Kq831PZT2yxGTgG_ka9sm3woOQp0SKKFTDwdSisBxTcrGkPAV0z4XqufjGT-5KB-A0_pFAUVSsvEZ2Kou5gaZe7W5NwiniuTvFhpvqRo520ye3of24A0ozpmvX6Tke74wW8PkrrvRc7FKRSgPo5jkLt2rDw7lSRhBKdihE0FjsktgV9EdCTnXsZSuVr9qZVIpXp46vvUI29QsucFmq3dI-LUHaELgQDEkhFmGbc50w0qGqYUItCHPyPbq63dTVSv0JlP0GTnd75zQ41URLI_BzzlJUbukgSUMLAd0hXxiOQ9xYLTQam7wUhw_ds05EQvUzT-ey8uKdJivdqtzSmId6vyc3P2NTGir309RwO4tJVmDwrSRYva1AdcP_ocu5CS1iocdq5KKt6svjrg9FFH29T9Hsjco6wJ05lwuqZESKsiiqwtGN83sV5DBZ8t1Ne0f2cPD9wD1DEIwvzV0yyfS9KQaOJWq9jYeIgJBR22eIbfv-m-GK8y961RHxSYftuzu0gzUlEp2KwJz2bM1me9ZeUuiNTGWHU5F6FKoDP1_stwSMetQ9EZJ7inlRZqsyDg_Vz-fmnsAYUSt_ozJfcVoOFymd6RnlDGtp37TRmqKulxbznbk3YyFmS8om3RVN36QItECY3zUFc5_YQ&isca=1)!

## Table of Contents
* [The Model](#model)
* [The WebApp](#webapp)
* [Installation](#installation)
* [Usage](#usage)
* [Credits](#credits)

## Model

The language model is based on GPT-2 with the [Pseudo-Self-Attention](https://arxiv.org/pdf/1908.06938.pdf) mechanism integrated in to allow it to generate based on news images and categories. A GPT-2 model was acquired that was pretrained on a large corpus of Filipino text. It was then fine-tuned on the BalitaNLP dataset for image and class conditional text generation. The model was then trained for 2 weeks on Cloud TPUs

Built with: Python, Pytorch, Huggingface

## WebApp

Generate articles with the webapp! Users can upload their own images and also start off the articles with their own titles and prompts. You can view token attention for each layer and head of the language model with color visualization. The output token probabilities for each token are also displayed in a graph.

Built with: Django, Vue

**Visualize attention of the GPT-2 model**

<img src="./readme_images/GeneratedArticle.JPG"/>

**View a graph of output token probabilities**

<img src="./readme_images/GeneratedArticleScores.png"/>

**You can upload your own image and give a title and prompt to start off the article OR you can use one of the sample images**

<img src="./readme_images/Form.png" width="65%" />

## Installation

1. Install [python 3.9](https://www.python.org/downloads/)

2. Install the requirements

`pip install -r requirements.txt`

## Usage
1. Download the model

`python download_model.py`

2. Navigate to the vue frontend folder

`cd project/frontend`

3. Build the vue project

`npm run build`

4. Navigate to project folder

`cd ../`

5. Run the django server

`python manage.py runserver`

6. Open the local development server url

## Credits

Big thanks to the TPU Research Cloud programfor allowing me to train on their cloud TPUs for free.