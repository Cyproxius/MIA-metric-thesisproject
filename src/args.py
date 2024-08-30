class ExperimentArgs:
  def __init__(self, target_model, unlearn_model, output_dir, key_name, data, length, save_loss):
      self.target_model = target_model
      self.unlearn_model = unlearn_model
      self.output_dir = output_dir
      self.key_name = key_name
      self.data = data
      self.length = length
      self.save_loss = save_loss

class UnlearningArgs:

  def __init__(self, lr, epochs, col_name, save_model, only_members, device):
    self.lr = lr
    self.epochs = epochs
    self.col_name = col_name
    self.save_model = save_model
    self.only_members = only_members
    self.device = device
    