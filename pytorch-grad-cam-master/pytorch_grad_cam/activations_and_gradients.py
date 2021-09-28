import torch
#
# def ints_to_tensor(ints):
#     """
#     Converts a nested list of integers to a padded tensor.
#     """
#     if isinstance(ints, torch.Tensor):
#         return ints
#     if isinstance(ints, list):
#         if isinstance(ints[0], int):
#             return torch.LongTensor(ints)
#         if isinstance(ints[0], torch.Tensor):
#             return pad_tensors(ints)
#         if isinstance(ints[0], list):
#             return ints_to_tensor([ints_to_tensor(inti) for inti in ints])
#
# def pad_tensors(tensors):
#     """
#     Takes a list of `N` M-dimensional tensors (M<4) and returns a padded tensor.
#
#     The padded tensor is `M+1` dimensional with size `N, S1, S2, ..., SM`
#     where `Si` is the maximum value of dimension `i` amongst all tensors.
#     """
#     rep = tensors[0]
#     padded_dim = []
#     for dim in range(rep.dim()):
#         max_dim = max([tensor.size(dim) for tensor in tensors])
#         padded_dim.append(max_dim)
#     padded_dim = [len(tensors)] + padded_dim
#     padded_tensor = torch.zeros(padded_dim)
#     padded_tensor = padded_tensor.type_as(rep)
#     for i, tensor in enumerate(tensors):
#         size = list(tensor.size())
#         if len(size) == 1:
#             padded_tensor[i, :size[0]] = tensor
#         elif len(size) == 2:
#             padded_tensor[i, :size[0], :size[1]] = tensor
#         elif len(size) == 3:
#             padded_tensor[i, :size[0], :size[1], :size[2]] = tensor
#         else:
#             raise ValueError('Padding is supported for upto 3D tensors at max.')
#     return padded_tensor
#

class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layer, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform

        target_layer.register_forward_hook(self.save_activation)

        #Backward compitability with older pytorch versions:
        if hasattr(target_layer, 'register_full_backward_hook'):
            target_layer.register_full_backward_hook(self.save_gradient)
        else:
            target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        activation = output
        # print(output[1].size())

        # target = output
        # max_cols = max([len(row) for batch in target for row in batch])
        # max_rows = max([len(batch) for batch in target])
        # padded = [batch + [[0] * (max_cols)] * (max_rows - len(batch)) for batch in target]
        # padded = torch.tensor([row + [0] * (max_length - len(row)) for batch in padded for row in batch])
        # activation = padded.view(-1, max_rows, max_cols)

        print('it is --',activation)
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)

        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, grad_input, grad_output):
        # Gradients are computed in reverse order
        grad = grad_output[0]
        if self.reshape_transform is not None:
            grad = self.reshape_transform(grad)
        self.gradients = [grad.cpu().detach()] + self.gradients

    def __call__(self, x):
        self.gradients = []
        self.activations = []

        return self.model(x)