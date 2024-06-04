import collections
import typing

import numpy as np


_field_names = [
    "state",
    "action",
    "reward",
    "next_state",
    "done"
]
Experience = collections.namedtuple("Experience", field_names=_field_names)


class PrioritizedExperienceReplayBuffer:
    """
    Fixed-size buffer to store priority, Experience tuples.

    Attributes:
        _batch_size (int): Number of experience samples per training batch.
        _buffer_size (int): Maximum number of prioritized experience tuples stored in buffer.
        _buffer_length (int): Current number of prioritized experience tuples in buffer.
        _buffer (np.array): Array to store priority and experience tuples.
        _alpha (float): Strength of prioritized sampling.
        _random_state (np.random.RandomState): Random number generator.
    """

    def __init__(self,
                 batch_size: int,
                 buffer_size: int,
                 alpha: float = 0.0,
                 random_state: np.random.RandomState = None) -> None:
        """
        Initialize an ExperienceReplayBuffer object.

        Args:
            batch_size (int): Size of each training batch.
            buffer_size (int): Maximum size of buffer.
            alpha (float): Strength of prioritized sampling. Default to 0.0 (i.e., uniform sampling).
            random_state (np.random.RandomState): Random number generator.
        """
        self._batch_size = batch_size
        self._buffer_size = buffer_size
        self._buffer_length = 0 # current number of prioritized experience tuples in buffer
        self._buffer = np.empty(self._buffer_size, dtype=[("priority", np.float32), ("experience", Experience)])
        self._alpha = alpha
        self._random_state = np.random.RandomState() if random_state is None else random_state
        
    def __len__(self) -> int:
        """
        Current number of prioritized experience tuple stored in buffer.

        Returns:
            int: Number of experience tuples stored in buffer.
        """
        return self._buffer_length

    @property
    def alpha(self):
        """
        Strength of prioritized sampling.

        Returns:
            float: Strength of prioritized sampling.
        """
        return self._alpha

    @property
    def batch_size(self) -> int:
        """
        Number of experience samples per training batch.

        Returns:
            int: Number of experience samples per training batch.
        """
        return self._batch_size
    
    @property
    def buffer_size(self) -> int:
        """
        Maximum number of prioritized experience tuples stored in buffer.

        Returns:
            int: Maximum number of prioritized experience tuples stored in buffer.
        """
        return self._buffer_size

    def add(self, experience: Experience) -> None:
        """
        Add a new experience to memory.

        Args:
            experience (Experience): Experience tuple to add to memory.
        """
        priority = 1.0 if self.is_empty() else self._buffer["priority"].max()
        if self.is_full():
            if priority > self._buffer["priority"].min():
                idx = self._buffer["priority"].argmin()
                self._buffer[idx] = (priority, experience)
            else:
                pass # low priority experiences should not be included in buffer
        else:
            self._buffer[self._buffer_length] = (priority, experience)
            self._buffer_length += 1

    def is_empty(self) -> bool:
        """
        Check if the buffer is empty.

        Returns:
            bool: True if the buffer is empty; False otherwise.
        """
        return self._buffer_length == 0
    
    def is_full(self) -> bool:
        """
        Check if the buffer is full.

        Returns:
            bool: True if the buffer is full; False otherwise.
        """
        return self._buffer_length == self._buffer_size
    
    def sample(self, beta: float) -> typing.Tuple[np.array, np.array, np.array]:
        """
        Sample a batch of experiences from memory.

        Args:
            beta (float): Beta parameter for importance sampling.

        Returns:
            typing.Tuple[np.array, np.array, np.array]: Tuple of sampled indices, experiences, and normalized weights.
        """
        # use sampling scheme to determine which experiences to use for learning
        ps = self._buffer[:self._buffer_length]["priority"]
        sampling_probs = ps**self._alpha / np.sum(ps**self._alpha)
        idxs = self._random_state.choice(np.arange(ps.size),
                                         size=self._batch_size,
                                         replace=True,
                                         p=sampling_probs)
        
        # select the experiences and compute sampling weights
        experiences = self._buffer["experience"][idxs]        
        weights = (self._buffer_length * sampling_probs[idxs])**-beta
        normalized_weights = weights / weights.max()
        
        return idxs, experiences, normalized_weights

    def update_priorities(self, idxs: np.array, priorities: np.array) -> None:
        """
        Update the priorities associated with particular experiences.

        Args:
            idxs (np.array): Indices of experiences to update.
            priorities (np.array): New priorities for the experiences.
        """
        self._buffer["priority"][idxs] = priorities