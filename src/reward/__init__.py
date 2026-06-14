"""Shared reward logic — used by both the attacker and the defender."""

from src.reward.reward_computer import RewardComputer, RewardConfig, RewardResult, TopicProfile
from src.reward.reward_function import (
    RewardFunction,
    RewardWeights,
    RewardOutput,
    FeatureExtractor,
    DistanceCalculator,
    SegmentFeatures,
    DistanceScores,
)
from src.reward.defender_reward import (
    DefenderRewardFunction,
    DefenderRewardOutput,
    DefenderRewardWeights,
)

__all__ = [
    "RewardComputer", "RewardConfig", "RewardResult", "TopicProfile",
    "RewardFunction", "RewardWeights", "RewardOutput",
    "FeatureExtractor", "DistanceCalculator", "SegmentFeatures", "DistanceScores",
    "DefenderRewardFunction", "DefenderRewardOutput", "DefenderRewardWeights",
]
