# -*- coding: UTF-8 -*- 

from mindformers.models.gpt2 import GPT2LMHeadModel, GPT2Tokenizer
from mindformers import Trainer
import json

model = GPT2LMHeadModel.from_pretrained('./gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('./gpt2')

trainer = Trainer(task='text_generation',                  
                  model=model,
                  train_dataset='data\\train.mindrecord',
                  tokenizer=tokenizer,
                  pet_method='lora')
trainer.finetune(finetune_checkpoint="{checkpoint file path}")
