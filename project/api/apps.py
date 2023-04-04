import os

from django.apps import AppConfig
from django.conf import settings

import torch
from transformers import GPT2TokenizerFast, GPT2Config

from .ML import psa_gpt2

class ApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'api'

    checkpoint_name = os.path.join(settings.BASE_DIR, 'api/ML/checkpoint-263820')
    config = GPT2Config.from_pretrained(checkpoint_name)
    tokenizer = GPT2TokenizerFast.from_pretrained(checkpoint_name)
    model = psa_gpt2.ImageCaptioning.from_pretrained(checkpoint_name, config=config)
    model.load_state_dict(torch.load(f'{checkpoint_name}/pytorch_model.bin'))
    model.to(settings.PYTORCH_DEVICE)
    model.eval()
