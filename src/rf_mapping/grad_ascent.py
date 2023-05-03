import torch
import torch.optim as optim

__all__ = ['GradientAscent']


class GradientAscent:
    def __init__(self, truncated_model: torch.nn.Module, unit_index: int, img: torch.Tensor,
                 lr: float = 0.1, optimizer: str = 'SGD', momentum: bool = False):
        """
        Performs gradient ascent on a given image to maximize the response of a specified unit in a neural network.
        
        Args:
            truncated_model: The truncated neural network.
            unit_index: The index of the unit of interest.
            img: The starting image for optimization.
            lr: The learning rate for the optimizer.
            optimizer: The optimizer to use. Options: 'SGD', 'Adam'.
            momentum: Whether to use momentum with the optimizer.
        """
        self.model = truncated_model
        self.unit_index = unit_index
        self.img = img.requires_grad_(True)
        self.optimizer = self._get_optimizer(optimizer, lr, momentum)

    def _get_optimizer(self, optimizer_name: str, lr: float, momentum: bool) -> optim.Optimizer:
        if optimizer_name == 'Adam':
            return optim.Adam([self.img], lr=lr)
        elif optimizer_name == 'SGD':
            return optim.SGD([self.img], lr=lr, momentum=momentum)
        else:
            raise ValueError(f'Optimizer "{optimizer_name}" not supported')

    def _objective_function(self, x: torch.Tensor) -> torch.Tensor:
        responses = self.model(x)
        num_images, num_units, ny, nx = responses.shape
        return responses[0, self.unit_index, ny//2, nx//2]

    def step(self) -> torch.Tensor:
        """
        Takes one optimization step and returns the updated image tensor.
        
        Returns:
            The updated image tensor.
        """
        self.optimizer.zero_grad()

        # Need to put a negative sign because optimizer minimizes the "loss",
        # but this is an response, and we want to maximize it.
        response = -self._objective_function(self.img)

        # Compute the gradient of the response with respect to the image.
        response.backward()

        # Update the image using the optimizer.
        self.optimizer.step()

        # Reset the gradient to zero.
        self.img.grad.zero_()

        return self.img
