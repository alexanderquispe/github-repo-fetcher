"""
Analysis modules for AI Tool Adoption Contagion Study.

This package provides utilities for analyzing contagion and cascade effects
in AI coding tool adoption through developer collaboration networks.

Modules:
    - adoption_timeline: Build per-user adoption timelines across tools
    - cascade_analysis: Detect and analyze adoption cascades
    - collaboration_network: Build weighted collaboration networks
    - contagion_analysis: Simplified contagion analysis (recommended)
    - contagion_models: Epidemic models (IC, LT, SIR)
    - influence_analysis: Peer influence and homophily analysis
    - network_builder: Basic network construction utilities
    - tool_switching: Markov model for tool transitions
    - user_matcher: Cross-tool user matching
"""

from .adoption_timeline import AdoptionTimelineBuilder
from .cascade_analysis import CascadeDetector, analyze_cascades
from .collaboration_network import CollaborationNetworkBuilder, build_full_collaboration_network
from .contagion_analysis import ContagionAnalyzer, run_contagion_analysis
from .contagion_models import IndependentCascadeModel, LinearThresholdModel, SIRModel
from .tool_switching import ToolSwitchingModel, analyze_tool_switching
from .user_matcher import CrossToolUserMatcher

__all__ = [
    "AdoptionTimelineBuilder",
    "CascadeDetector",
    "analyze_cascades",
    "CollaborationNetworkBuilder",
    "build_full_collaboration_network",
    "ContagionAnalyzer",
    "run_contagion_analysis",
    "IndependentCascadeModel",
    "LinearThresholdModel",
    "SIRModel",
    "ToolSwitchingModel",
    "analyze_tool_switching",
    "CrossToolUserMatcher",
]
