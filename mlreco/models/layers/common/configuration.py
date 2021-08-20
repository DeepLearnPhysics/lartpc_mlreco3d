def setup_cnn_configuration(self, cfg, name):
    '''
    Base function for global network parameters.
    '''
    model_cfg = cfg.get(name, {})
    # Dimension of dataset
    self.D = model_cfg.get('data_dim', 3)
    # Number of input data features
    self.num_input = model_cfg.get('num_input', 1)
    # Allow biases in convolutions and linear layers
    self.allow_bias = model_cfg.get('allow_bias', True)
    # Spatial size of dataset
    self.spatial_size = model_cfg.get('spatial_size', 512)

    # Define activation function
    self.leakiness = model_cfg.get('leakiness', 0.33)
    self.activation_cfg = model_cfg.get('activation', {})
    self.activation_name = self.activation_cfg.get('name', 'lrelu')
    self.activation_args = self.activation_cfg.get(
        'args', {})

    # Define normalization function
    # print(model_cfg)
    self.norm_cfg = model_cfg.get('norm_layer', {})
    self.norm = self.norm_cfg.get('name', 'batch_norm')
    self.norm_args = self.norm_cfg.get('args', {})
