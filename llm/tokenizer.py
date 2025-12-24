import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken

class GPTDataSetV1(Dataset):

  def __init__(self, text, stride, tokenizer, max_length) -> None:
    self.input_ids = []
    self.target_ids = []

    token_ids = tokenizer.encode(text)

    for i in range(0, len(token_ids) - max_length, stride):
      input_chunk = token_ids[i : i + max_length] # Splits the text into chunks of size max_length
      target_chunk = token_ids[i + 1 : i + max_length + 1] # Splits the text into chunks of size max_length + 1
      self.input_ids.append(torch.Tensor(input_chunk)) # Appends the input chunk to the input_ids list
      self.target_ids.append(torch.Tensor(target_chunk)) # Appends the target chunk to the target_ids list

  def __len__(self):
    return len(self.input_ids) # Returns the length of the input_ids list

  def __getitem__(self, idx):
    return self.input_ids[idx], self.target_ids[idx] # Returns the input_ids and target_ids at the given index


def create_dataloader_v1(
    text, batch_size = 4, max_length = 256, stride = 128, shuffle = True, drop_last = True, num_workers = 0
    ):
  tokenizer = tiktoken.get_encoding("gpt2") # Gets the gpt2 tokenizer
  dataset = GPTDataSetV1(text, stride, tokenizer, max_length) # Creates the dataset
  dataloader = DataLoader(
      dataset, batch_size = batch_size, shuffle = shuffle, drop_last = drop_last, num_workers=num_workers
      ) # Creates the dataloader
  return dataloader