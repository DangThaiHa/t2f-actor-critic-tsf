### External datasets used can be manually downloaded from open sources:

# Household Power Consumption (HPC)

{
  author       = {Hebrail, Georges and Berard, Alice},
  title        = {{Individual Household Electric Power Consumption}},
  year         = {2006},
  howpublished = {UCI Machine Learning Repository},
  note         = {{DOI}: https://doi.org/10.24432/C58K54}
}

# Electricity Load Diagrams (ELD)

{
  author       = {Trindade, Artur},
  title        = {ElectricityLoadDiagrams20112014},
  year         = {2015},
  howpublished = {UCI Machine Learning Repository},
  note         = {DOI: 10.24432/C58C86}
}

# Electricity Transformer Temperature datasets ETT-h (hourly) and ETT-m (15 minute)

{
  title={Informer: Beyond efficient transformer for long sequence time-series forecasting},
  author={Zhou, Haoyi and Zhang, Shanghang and Peng, Jieqi and Zhang, Shuai and Li, Jianxin and Xiong, Hui and Zhang, Wancai},
  booktitle={Proceedings of AAAI},
  year={2021}
}

After downloading the datasets, pass them in a single index tabular format to the preprocessing batch process (example [etth/bash.sh]) to ensure the format and structure meet the requirements.

The dummy set generation steps can be found and adapted in [dummy/dummy_generator].