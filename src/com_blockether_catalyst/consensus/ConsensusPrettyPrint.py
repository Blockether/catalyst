"""
Pretty printing utilities for consensus rounds with tables and similarity scores.

This module provides informative table-based logging for consensus rounds,
showing model responses, similarity scores, and field-by-field comparisons.
"""

import json
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Union
from io import StringIO

from pydantic import BaseModel, RootModel

from .ConsensusTypes import (
    ConsensusRound,
    DisagreementAnalysis,
    ModelResponse,
    VerbosityLevel,
)


class ConsensusPrettyPrinter:
    """Pretty printer for consensus rounds.

    All methods return formatted strings instead of printing directly
    to avoid output interleaving in parallel execution.
    """

    # Class constants to avoid magic numbers
    _MAX_STRING_DISPLAY_LENGTH: int = 200
    _MAX_STRING_DISPLAY_SHORT: int = 50
    _MAX_DICT_DISPLAY_LENGTH: int = 500
    _MAX_LIST_ITEMS_DISPLAY: int = 10
    _DEFAULT_MAX_FIELDS_NORMAL: int = 4
    _VOTE_HASH_DISPLAY_LENGTH: int = 8
    _PROMPT_PREVIEW_LENGTH: int = 200
    _FIELD_SUMMARY_LENGTH: int = 30
    _VALUE_TRUNCATE_LENGTH: int = 100
    _SEPARATOR_WIDTH: int = 80

    @staticmethod
    def format_round_summary(
        round_num: int,
        round_data: ConsensusRound,
        query: str,
        verbosity: VerbosityLevel = VerbosityLevel.NORMAL,
        max_fields_normal: int = 4,
        prompts: Optional[Dict[str, str]] = None,
    ) -> str:
        """Format a summary of a consensus round.

        Args:
            round_num: The round number
            round_data: The consensus round data
            query: The original query
            verbosity: Verbosity level for field display
            max_fields_normal: Maximum fields to show in NORMAL mode
            prompts: Optional dict mapping model IDs to their specific prompts

        Returns:
            Formatted string representation of the round
        """
        result = []

        # Header
        result.append("\n" + "=" * ConsensusPrettyPrinter._SEPARATOR_WIDTH)
        result.append(f"🔄 CONSENSUS ROUND {round_num}")
        result.append("=" * ConsensusPrettyPrinter._SEPARATOR_WIDTH)

        # Query section
        result.append("\n💭 INPUT QUERY:")
        result.append("-" * 40)
        result.append(query)

        # Model-specific prompts (if verbose)
        if prompts and verbosity.value >= VerbosityLevel.VERBOSE.value:
            result.append("\n📨 MODEL-SPECIFIC PROMPTS:")
            result.append("-" * 40)
            for model_id, prompt in prompts.items():
                prompt_display = (
                    prompt[:ConsensusPrettyPrinter._PROMPT_PREVIEW_LENGTH] + "..."
                    if len(prompt) > ConsensusPrettyPrinter._PROMPT_PREVIEW_LENGTH
                    else prompt
                )
                result.append(f"\n{model_id}:")
                result.append(f"  {prompt_display}")

        # Calculate vote groups
        vote_groups = ConsensusPrettyPrinter._get_vote_groups(round_data.responses)
        total_models = len(round_data.responses)

        # Model responses section
        result.append("\n🤖 MODEL RESPONSES:")
        result.append("=" * ConsensusPrettyPrinter._SEPARATOR_WIDTH)

        for response in round_data.responses:
            model_id = response.id
            vote_hash = ConsensusPrettyPrinter._get_vote_hash(response)

            # Determine consensus status
            group_size = vote_groups.get(vote_hash, 1)
            consensus_status = ConsensusPrettyPrinter._format_consensus_status_plain(
                group_size, total_models
            )

            result.append(f"\n[{model_id}]")
            result.append(f"Vote Hash: {vote_hash[:ConsensusPrettyPrinter._VOTE_HASH_DISPLAY_LENGTH]}")
            result.append(f"Status: {consensus_status}")
            result.append("-" * 40)

            if not isinstance(response.content, (BaseModel, RootModel)):
                raise ValueError("Response content must be a Pydantic BaseModel or RootModel")

            content_dict: Dict[str, Any] = response.content.model_dump()
            reasoning: Optional[str] = content_dict.pop("reasoning", None)

            # Show fields based on verbosity
            field_count = 0
            for key, value in content_dict.items():
                field_count += 1

                # In NORMAL mode, limit fields shown
                if verbosity == VerbosityLevel.NORMAL and field_count > max_fields_normal:
                    remaining = len(content_dict) - max_fields_normal
                    if remaining > 0:
                        result.append(f"  ... and {remaining} more fields")
                    break

                # Format value based on type and verbosity
                value_str = ConsensusPrettyPrinter._format_value(value, verbosity)
                result.append(f"  {key}: {value_str}")

            # Add reasoning if in verbose mode
            if verbosity == VerbosityLevel.VERBOSE and reasoning:
                result.append("\n  Reasoning:")
                if len(reasoning) > ConsensusPrettyPrinter._MAX_STRING_DISPLAY_LENGTH:
                    reasoning = reasoning[:ConsensusPrettyPrinter._MAX_STRING_DISPLAY_LENGTH] + "..."
                result.append(f"    {reasoning}")

        # Disagreement analysis if available
        if round_data.disagreement_analysis:
            result.append(ConsensusPrettyPrinter._format_disagreement_analysis(
                round_data.disagreement_analysis, verbosity
            ))

        result.append("\n" + "=" * ConsensusPrettyPrinter._SEPARATOR_WIDTH + "\n")
        return "\n".join(result)

    @staticmethod
    def format_consensus_achieved(
        round_num: int,
        num_responses: int,
        duration_ms: Optional[float] = None,
    ) -> str:
        """Format a success message for consensus achievement.

        Args:
            round_num: The round number where consensus was achieved
            num_responses: Number of model responses
            duration_ms: Optional duration in milliseconds

        Returns:
            Formatted success message string
        """
        result = []
        result.append("\n" + "=" * ConsensusPrettyPrinter._SEPARATOR_WIDTH)
        result.append("✅ CONSENSUS ACHIEVED!")
        
        status_line = f"Round: {round_num} | Models: {num_responses}"
        if duration_ms:
            status_line += f" | Duration: {duration_ms:.1f}ms"
        result.append(status_line)

        result.append("=" * ConsensusPrettyPrinter._SEPARATOR_WIDTH + "\n")
        return "\n".join(result)

    @staticmethod
    def format_round_comparison(
        rounds: List[ConsensusRound],
        verbosity: VerbosityLevel = VerbosityLevel.NORMAL,
    ) -> str:
        """Format a comparison table showing evolution across rounds.

        Args:
            rounds: List of consensus rounds
            verbosity: Verbosity level for display

        Returns:
            Formatted comparison table string
        """
        if len(rounds) < 2:
            return ""

        result = []
        result.append("\n📊 CONSENSUS EVOLUTION ACROSS ROUNDS")
        result.append("=" * ConsensusPrettyPrinter._SEPARATOR_WIDTH)

        # Table header
        header_format = "{:<8} {:<8} {:<15} {:<12} {:<25} {:<25}"
        result.append(header_format.format(
            "Round", "Models", "Unique Votes", "Agreement %", "Consensus Fields", "Disputed Fields"
        ))
        result.append("-" * ConsensusPrettyPrinter._SEPARATOR_WIDTH)

        # Table rows
        for round_data in rounds:
            # Calculate metrics
            num_models: int = len(round_data.responses)
            unique_votes: int = len(ConsensusPrettyPrinter._get_unique_votes(round_data.responses))
            agreement_pct: float = ((num_models - unique_votes + 1) / num_models * 100) if num_models > 0 else 0

            # Get field info
            consensus_fields: List[str] = []
            disputed_fields: List[str] = []

            if round_data.disagreement_analysis:
                consensus_fields = round_data.disagreement_analysis.consensus_fields[:3]
                disputed_fields = list(round_data.disagreement_analysis.disagreement_fields.keys())[:3]

            row = header_format.format(
                str(round_data.round_number),
                str(num_models),
                str(unique_votes),
                f"{agreement_pct:.1f}%",
                ", ".join(consensus_fields) if consensus_fields else "—",
                ", ".join(disputed_fields) if disputed_fields else "—",
            )
            result.append(row)

        result.append("=" * ConsensusPrettyPrinter._SEPARATOR_WIDTH + "\n")
        return "\n".join(result)

    @staticmethod
    def _format_value(value: Any, verbosity: VerbosityLevel) -> str:
        """Format a value for display based on its type and verbosity.

        Args:
            value: The value to format
            verbosity: Current verbosity level

        Returns:
            Formatted string representation
        """
        if isinstance(value, (list, dict)):
            if isinstance(value, list):
                if verbosity == VerbosityLevel.VERBOSE and len(value) <= ConsensusPrettyPrinter._MAX_LIST_ITEMS_DISPLAY:
                    # Show list items in verbose mode if not too many
                    items_str = []
                    for i, item in enumerate(value[:ConsensusPrettyPrinter._MAX_LIST_ITEMS_DISPLAY]):
                        item_str = json.dumps(item) if not isinstance(item, str) else f'"{item}"'
                        if len(item_str) > ConsensusPrettyPrinter._MAX_STRING_DISPLAY_SHORT:
                            item_str = item_str[:ConsensusPrettyPrinter._MAX_STRING_DISPLAY_SHORT - 3] + "..."
                        items_str.append(item_str)
                    return f"[{', '.join(items_str)}]"
                else:
                    return f"[{len(value)} items]"
            else:
                return f"{{{len(value)} fields}}"
        elif isinstance(value, str):
            max_len = (ConsensusPrettyPrinter._MAX_STRING_DISPLAY_LENGTH
                      if verbosity == VerbosityLevel.VERBOSE
                      else ConsensusPrettyPrinter._MAX_STRING_DISPLAY_SHORT)
            if len(value) <= max_len:
                return f'"{value}"'
            else:
                return f'"{value[:max_len - 3]}..."'
        else:
            return json.dumps(value)

    @staticmethod
    def _format_consensus_status_plain(group_size: int, total_models: int) -> str:
        """Format the consensus status for a model based on its vote group.

        Args:
            group_size: Number of models with the same vote
            total_models: Total number of models

        Returns:
            Plain text consensus status string
        """
        if group_size == total_models:
            return f"✓ Consensus ({group_size}/{total_models})"
        elif group_size > total_models / 2:
            return f"✓ Majority ({group_size}/{total_models})"
        elif group_size > 1:
            return f"⚠ Minority ({group_size}/{total_models})"
        else:
            return "✗ Outlier (1/1)"

    @staticmethod
    def _format_disagreement_analysis(
        analysis: DisagreementAnalysis,
        verbosity: VerbosityLevel,
    ) -> str:
        """Format disagreement analysis.

        Args:
            analysis: Disagreement analysis
            verbosity: Verbosity level

        Returns:
            Formatted disagreement analysis string
        """
        if not analysis.disagreement_fields:
            return ""

        result = []
        result.append("\n⚠️  FIELD DISAGREEMENTS:")
        result.append("-" * ConsensusPrettyPrinter._SEPARATOR_WIDTH)

        # Table header
        header_format = "{:<30} {:<15} {}"
        result.append(header_format.format("Field", "# Values", "Examples"))
        result.append("-" * ConsensusPrettyPrinter._SEPARATOR_WIDTH)

        max_fields: int = 10 if verbosity == VerbosityLevel.VERBOSE else 5

        for field, values in list(analysis.disagreement_fields.items())[:max_fields]:
            unique_values: List[str] = list(set(str(v) for v in values))
            num_different: int = len(unique_values)

            # Format examples
            examples = []
            show_count = 3 if num_different <= 3 else 2
            for v_str in unique_values[:show_count]:
                if len(v_str) > 30:
                    v_str = v_str[:27] + "..."
                examples.append(v_str)

            if num_different > show_count:
                examples.append(f"... +{num_different - show_count} more")

            row = header_format.format(
                field if len(field) <= 30 else field[:27] + "...",
                str(num_different),
                ", ".join(examples)
            )
            result.append(row)

        if len(analysis.disagreement_fields) > max_fields:
            remaining = len(analysis.disagreement_fields) - max_fields
            result.append(f"\n... and {remaining} more fields with disagreements")

        # Summary if there are many disagreements
        if len(analysis.disagreement_fields) > 3:
            result.append("\n" + "-" * ConsensusPrettyPrinter._SEPARATOR_WIDTH)
            result.append("⚠️  High Disagreement Detected")
            result.append(f"Total fields with disagreements: {len(analysis.disagreement_fields)}")
            if verbosity == VerbosityLevel.NORMAL:
                result.append("(Use verbose mode to see more details)")

        return "\n".join(result)

    @staticmethod
    def _calculate_similarity_scores(
        responses: List[ModelResponse],
    ) -> Dict[str, float]:
        """Calculate average similarity scores for each model response.

        Args:
            responses: List of model responses

        Returns:
            Dictionary mapping model ID to average similarity score
        """
        scores: Dict[str, float] = {}

        for i, response1 in enumerate(responses):
            similarities: List[float] = []
            for j, response2 in enumerate(responses):
                if i != j:
                    similarity = ConsensusPrettyPrinter._calculate_response_similarity(response1, response2)
                    similarities.append(similarity)

            if similarities:
                scores[response1.id] = sum(similarities) / len(similarities)
            else:
                scores[response1.id] = 1.0

        return scores

    @staticmethod
    def _calculate_response_similarity(response1: ModelResponse, response2: ModelResponse) -> float:
        """Calculate similarity between two responses.

        Args:
            response1: First response
            response2: Second response

        Returns:
            Similarity score between 0 and 1
        """
        from .ConsensusTypes import ComparisonStrategy

        # Get response content as dicts
        dict1: Dict[str, Any] = response1.content.model_dump()
        dict2: Dict[str, Any] = response2.content.model_dump()

        # Get the model class to check field metadata
        model_class = response1.content.__class__

        # Count matching fields (excluding fields with IGNORE strategy)
        matching_fields: int = 0
        total_fields: int = 0

        for field_name in dict1:
            # Check if field should be ignored based on VotingField metadata
            field_info = model_class.model_fields.get(field_name)

            # Check if field has IGNORE strategy
            if field_info and field_info.json_schema_extra:
                extra = field_info.json_schema_extra
                if isinstance(extra, dict) and "voting_comparison" in extra:
                    voting_comparison = extra["voting_comparison"]
                    if isinstance(voting_comparison, dict):
                        strategy = voting_comparison.get("strategy")
                        # Check both enum and string values
                        if (
                            (isinstance(strategy, ComparisonStrategy) and strategy == ComparisonStrategy.IGNORE)
                            or (isinstance(strategy, str) and strategy == "ignore")
                            or strategy == "IGNORE"
                        ):
                            continue  # Skip ignored fields

            total_fields += 1
            if field_name in dict2 and dict1[field_name] == dict2[field_name]:
                matching_fields += 1

        return matching_fields / total_fields if total_fields > 0 else 0.0

    @staticmethod
    def _get_vote_hash(response: ModelResponse) -> str:
        """Get the voting hash for a response.

        Args:
            response: Model response

        Returns:
            Vote hash string
        """
        vote_key: Any = response.content.get_voting_key()
        return str(vote_key)

    @staticmethod
    def _get_unique_votes(responses: List[ModelResponse]) -> Set[str]:
        """Get unique vote hashes from responses.

        Args:
            responses: List of model responses

        Returns:
            Set of unique vote hashes
        """
        return {ConsensusPrettyPrinter._get_vote_hash(response) for response in responses}

    @staticmethod
    def _get_vote_groups(responses: List[ModelResponse]) -> Dict[str, int]:
        """Get vote groups showing how many models voted for each option.

        Args:
            responses: List of model responses

        Returns:
            Dictionary mapping vote hash to count of models with that vote
        """
        vote_counts: Dict[str, int] = {}
        for response in responses:
            vote_hash = ConsensusPrettyPrinter._get_vote_hash(response)
            vote_counts[vote_hash] = vote_counts.get(vote_hash, 0) + 1
        return vote_counts

    @staticmethod
    def _get_response_summary(
        response: ModelResponse,
        verbosity: VerbosityLevel,
        max_fields: int = 4,
    ) -> str:
        """Get a summary of the response based on verbosity.

        Args:
            response: Model response
            verbosity: Verbosity level
            max_fields: Maximum fields to show in NORMAL mode

        Returns:
            Formatted response summary
        """
        content_dict: Dict[str, Any] = response.content.model_dump()

        # Remove reasoning field as it's usually too long
        content_dict.pop("reasoning", None)

        if verbosity == VerbosityLevel.VERBOSE:
            # Show all fields
            return ConsensusPrettyPrinter._format_dict_compact(content_dict)
        else:
            # Show only first few fields
            limited_dict: Dict[str, Any] = dict(list(content_dict.items())[:max_fields])
            if len(content_dict) > max_fields:
                limited_dict["..."] = f"({len(content_dict) - max_fields} more fields)"
            return ConsensusPrettyPrinter._format_dict_compact(limited_dict)

    @staticmethod
    def _format_dict_compact(d: Dict[str, Any]) -> str:
        """Format a dictionary in a compact way for display.

        Args:
            d: Dictionary to format

        Returns:
            Compact string representation
        """
        items: List[str] = []
        for key, value in d.items():
            if isinstance(value, (list, dict)):
                if isinstance(value, list):
                    value_str = f"[{len(value)} items]"
                else:
                    value_str = f"{{{len(value)} fields}}"
            elif isinstance(value, str) and len(value) > ConsensusPrettyPrinter._FIELD_SUMMARY_LENGTH:
                value_str = f'"{value[:ConsensusPrettyPrinter._FIELD_SUMMARY_LENGTH - 3]}..."'
            else:
                value_str = json.dumps(value)
            items.append(f"{key}: {value_str}")

        return "\n".join(items)

    # Backward compatibility - old methods now just print the formatted output
    @staticmethod
    def print_round_summary(
        round_num: int,
        round_data: ConsensusRound,
        query: str,
        verbosity: VerbosityLevel = VerbosityLevel.NORMAL,
        max_fields_normal: int = 4,
        prompts: Optional[Dict[str, str]] = None,
    ) -> None:
        """Print a summary of a consensus round (backward compatibility).

        This method prints immediately. For buffered output, use format_round_summary.
        """
        output = ConsensusPrettyPrinter.format_round_summary(
            round_num, round_data, query, verbosity, max_fields_normal, prompts
        )
        print(output, end='')

    @staticmethod
    def print_consensus_achieved(
        round_num: int,
        num_responses: int,
        duration_ms: Optional[float] = None,
    ) -> None:
        """Print a success message for consensus achievement (backward compatibility).

        This method prints immediately. For buffered output, use format_consensus_achieved.
        """
        output = ConsensusPrettyPrinter.format_consensus_achieved(round_num, num_responses, duration_ms)
        print(output, end='')

    @staticmethod
    def print_round_comparison(
        rounds: List[ConsensusRound],
        verbosity: VerbosityLevel = VerbosityLevel.NORMAL,
    ) -> None:
        """Print a comparison table showing evolution across rounds (backward compatibility).

        This method prints immediately. For buffered output, use format_round_comparison.
        """
        output = ConsensusPrettyPrinter.format_round_comparison(rounds, verbosity)
        if output:
            print(output, end='')
