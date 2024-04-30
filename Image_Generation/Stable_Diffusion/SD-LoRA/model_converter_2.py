def converter_model(state_dict):
    """
    Remove the 'module.' prefix from the state dictionary keys.
    This is common when models are trained using DataParallel.
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        if key == 'epoch': 
            new_state_dict[key] = state_dict['epoch']
            continue
        new_value_dict = {}
        for key_, value_ in value.items():
            new_key = key_.replace("module.", "")
            new_value_dict[new_key] = value_
        new_state_dict[key] = new_value_dict
    return new_state_dict
