import torch


def reached_waypoint(env, command_name: str = "base_velocity") -> torch.Tensor:
    return env.command_manager.get_term(command_name).just_reached.float()


def completed(env, command_name: str = "base_velocity") -> torch.Tensor:
    return env.command_manager.get_term(command_name).completed


def failed_termination(env, command_name: str = "base_velocity") -> torch.Tensor:
    """Penalize falls, but never penalize reaching the 400 m finish."""
    return (env.termination_manager.terminated & ~completed(env, command_name)).float()


def completion_bonus(env, command_name: str = "base_velocity") -> torch.Tensor:
    """One-step success bonus when the final ordered point is passed."""
    command = env.command_manager.get_term(command_name)
    return (command.completed & command.just_reached).float()
