"""
Output formatting utilities for consensus results.

This module provides comprehensive formatted output for consensus processes,
showing all rounds, model responses, metrics, and evolution.
"""

import json
import textwrap
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel

from .ConsensusTypes import (
    ConsensusResult,
    ConsensusRound,
    DisagreementAnalysis,
    ModelResponse,
    VerbosityLevel,
)


class ConsensusOutput:
    """Output formatter for consensus results.

    Provides a single method to format the complete consensus process output,
    including all rounds, responses, metrics, and evolution.
    """

    # Class constants to avoid magic numbers
    _MAX_STRING_DISPLAY_LENGTH: int = 200
    _MAX_STRING_DISPLAY_SHORT: int = 50
    _MAX_DICT_DISPLAY_LENGTH: int = 500
    _MAX_LIST_ITEMS_DISPLAY: int = 10
    _DEFAULT_MAX_FIELDS_VERBOSE: int = 4
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
        verbosity: VerbosityLevel = VerbosityLevel.VERBOSE,
        max_fields_normal: int = 4,
        prompts: Optional[Dict[str, str]] = None,
    ) -> str:
        """Format a summary of a consensus round.

        Args:
            round_num: The round number
            round_data: The consensus round data
            query: The original query
            verbosity: Verbosity level for field display
            max_fields_normal: Maximum fields to show in VERBOSE mode
            prompts: Optional dict mapping model IDs to their specific prompts

        Returns:
            Formatted string representation of the round
        """
        result = []

        # Header
        result.append("\n" + "=" * ConsensusOutput._SEPARATOR_WIDTH)
        result.append(f"🔄 CONSENSUS ROUND {round_num}")
        result.append("=" * ConsensusOutput._SEPARATOR_WIDTH)

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
                    prompt[: ConsensusOutput._PROMPT_PREVIEW_LENGTH] + "..."
                    if len(prompt) > ConsensusOutput._PROMPT_PREVIEW_LENGTH
                    else prompt
                )
                result.append(f"\n{model_id}:")
                result.append(f"  {prompt_display}")

        # Calculate vote groups (uses stored groups if available)
        vote_groups = ConsensusOutput._get_vote_groups_from_round(round_data)
        total_models = len(round_data.responses)

        # Model responses section
        result.append("\n🤖 MODEL RESPONSES:")
        result.append("=" * ConsensusOutput._SEPARATOR_WIDTH)

        for response in round_data.responses:
            model_id = response.id
            vote_hash = ConsensusOutput._get_response_vote_key(response, round_data)

            # Determine consensus status
            group_size = vote_groups.get(vote_hash, 1)
            consensus_status = ConsensusOutput._format_consensus_status_plain(group_size, total_models)

            result.append(f"\n[{model_id}]")
            result.append(f"Vote Hash: {vote_hash[: ConsensusOutput._VOTE_HASH_DISPLAY_LENGTH]}")
            result.append(f"Status: {consensus_status}")
            result.append("-" * 40)

            if not isinstance(response.content, BaseModel):
                raise ValueError("Response content must be a Pydantic BaseModel")

            content_dict: Dict[str, Any] = response.content.model_dump()
            reasoning: Optional[str] = content_dict.pop("reasoning", None)

            # Show fields based on verbosity
            field_count = 0
            for key, value in content_dict.items():
                field_count += 1

                # In VERBOSE mode, limit fields shown
                if verbosity == VerbosityLevel.VERBOSE and field_count > max_fields_normal:
                    remaining = len(content_dict) - max_fields_normal
                    if remaining > 0:
                        result.append(f"  ... and {remaining} more fields")
                    break

                # Format value based on type and verbosity
                value_str = ConsensusOutput._format_value(value, verbosity)
                result.append(f"  {key}: {value_str}")

            # Add reasoning if in verbose mode
            if verbosity == VerbosityLevel.VERBOSE and reasoning:
                result.append("\n  Reasoning:")
                if len(reasoning) > ConsensusOutput._MAX_STRING_DISPLAY_LENGTH:
                    reasoning = reasoning[: ConsensusOutput._MAX_STRING_DISPLAY_LENGTH] + "..."
                result.append(f"    {reasoning}")

        # Disagreement analysis if available
        if round_data.disagreement_analysis:
            result.append(ConsensusOutput._format_disagreement_analysis(round_data.disagreement_analysis, verbosity))

        result.append("\n" + "=" * ConsensusOutput._SEPARATOR_WIDTH + "\n")
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
        result.append("\n" + "=" * ConsensusOutput._SEPARATOR_WIDTH)
        result.append("✅ CONSENSUS ACHIEVED!")

        status_line = f"Round: {round_num} | Models: {num_responses}"
        if duration_ms:
            status_line += f" | Duration: {duration_ms:.1f}ms"
        result.append(status_line)

        result.append("=" * ConsensusOutput._SEPARATOR_WIDTH + "\n")
        return "\n".join(result)

    @staticmethod
    def format_round_comparison(
        rounds: List[ConsensusRound],
        verbosity: VerbosityLevel = VerbosityLevel.VERBOSE,
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
        result.append("=" * ConsensusOutput._SEPARATOR_WIDTH)

        # Table header
        header_format = "{:<8} {:<8} {:<15} {:<12} {:<25} {:<25}"
        result.append(
            header_format.format(
                "Round",
                "Models",
                "Unique Votes",
                "Agreement %",
                "Consensus Fields",
                "Disputed Fields",
            )
        )
        result.append("-" * ConsensusOutput._SEPARATOR_WIDTH)

        # Table rows
        for round_data in rounds:
            # Calculate metrics
            num_models: int = len(round_data.responses)
            unique_votes: int = len(ConsensusOutput._get_unique_votes(round_data.responses))
            # Calculate agreement with special handling for complete disagreement
            # - All models agree = 1.0
            # - All models disagree (each votes differently) = 0.0
            # - Partial agreement = largest_group / total_models
            #   e.g., 3 models where 2 agree = 2/3 = 0.667
            agreement: float
            if num_models == 0:
                agreement = 0.0
            elif unique_votes == 1:
                agreement = 1.0  # All models agree
            elif unique_votes == num_models:
                agreement = 0.0  # All models completely disagree
            else:
                # Partial agreement: use largest group fraction
                vote_groups = ConsensusOutput._get_vote_groups_from_round(round_data)
                max_group_size = max(vote_groups.values()) if vote_groups else 1
                agreement = max_group_size / num_models

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
                f"{agreement * 100:.1f}%",
                ", ".join(consensus_fields) if consensus_fields else "—",
                ", ".join(disputed_fields) if disputed_fields else "—",
            )
            result.append(row)

        result.append("=" * ConsensusOutput._SEPARATOR_WIDTH + "\n")
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
                if verbosity == VerbosityLevel.VERBOSE and len(value) <= ConsensusOutput._MAX_LIST_ITEMS_DISPLAY:
                    # Show list items in verbose mode if not too many
                    items_str = []
                    for i, item in enumerate(value[: ConsensusOutput._MAX_LIST_ITEMS_DISPLAY]):
                        item_str = json.dumps(item) if not isinstance(item, str) else f'"{item}"'
                        if len(item_str) > ConsensusOutput._MAX_STRING_DISPLAY_SHORT:
                            item_str = item_str[: ConsensusOutput._MAX_STRING_DISPLAY_SHORT - 3] + "..."
                        items_str.append(item_str)
                    return f"[{', '.join(items_str)}]"
                else:
                    return f"[{len(value)} items]"
            else:
                return f"{{{len(value)} fields}}"
        elif isinstance(value, str):
            max_len = (
                ConsensusOutput._MAX_STRING_DISPLAY_LENGTH
                if verbosity == VerbosityLevel.VERBOSE
                else ConsensusOutput._MAX_STRING_DISPLAY_SHORT
            )
            if len(value) <= max_len:
                return f'"{value}"'
            else:
                return f'"{value[: max_len - 3]}..."'
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
            return f"✗ Outlier ({group_size}/{total_models})"

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
        result.append("-" * ConsensusOutput._SEPARATOR_WIDTH)

        # Table header
        header_format = "{:<30} {:<15} {}"
        result.append(header_format.format("Field", "# Values", "Examples"))
        result.append("-" * ConsensusOutput._SEPARATOR_WIDTH)

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
                ", ".join(examples),
            )
            result.append(row)

        if len(analysis.disagreement_fields) > max_fields:
            remaining = len(analysis.disagreement_fields) - max_fields
            result.append(f"\n... and {remaining} more fields with disagreements")

        # Summary if there are many disagreements
        if len(analysis.disagreement_fields) > 3:
            result.append("\n" + "-" * ConsensusOutput._SEPARATOR_WIDTH)
            result.append("⚠️  High Disagreement Detected")
            result.append(f"Total fields with disagreements: {len(analysis.disagreement_fields)}")
            if verbosity == VerbosityLevel.VERBOSE:
                result.append("(Use verbose mode to see more details)")

        return "\n".join(result)

    @staticmethod
    def _get_vote_hash(response: ModelResponse) -> str:
        """Get the voting hash for a response.

        Args:
            response: Model response

        Returns:
            Vote hash string
        """
        # Use proper response grouping mechanism
        from ..consensus.Consensus import Consensus

        consensus_instance = Consensus.__new__(Consensus)  # Create instance without __init__
        # Pass empty cache and groups as parameters since they're no longer instance attributes
        vote_key: str = consensus_instance._get_voting_group(response.content, {}, [])
        return vote_key

    @staticmethod
    def _get_unique_votes(responses: List[ModelResponse]) -> Set[str]:
        """Get unique vote hashes from responses.

        Args:
            responses: List of model responses

        Returns:
            Set of unique vote hashes
        """
        return {ConsensusOutput._get_vote_hash(response) for response in responses}

    @staticmethod
    def _get_response_vote_key(response, round_data) -> str:
        """Get the vote key for a response, using stored groups if available.

        Args:
            response: The model response
            round_data: ConsensusRound containing responses and possibly stored vote_groups

        Returns:
            The vote key (either group key from stored groups or computed hash)
        """
        # If vote_groups are stored, find which group this response belongs to
        if hasattr(round_data, "vote_groups") and round_data.vote_groups:
            for group_key, group_responses in round_data.vote_groups.items():
                if any(r.id == response.id for r in group_responses):
                    return group_key

        # Fallback to computing hash
        return ConsensusOutput._get_vote_hash(response)

    @staticmethod
    def _get_vote_groups_from_round(round_data) -> Dict[str, int]:
        """Get vote groups from round data, using stored groups if available.

        Args:
            round_data: ConsensusRound containing responses and possibly stored vote_groups

        Returns:
            Dictionary mapping vote hash/group key to count of models with that vote
        """
        # If vote_groups are stored in the round, use them
        if hasattr(round_data, "vote_groups") and round_data.vote_groups:
            # Convert from Dict[str, List[ModelResponse]] to Dict[str, int]
            return {group_key: len(responses) for group_key, responses in round_data.vote_groups.items()}

        # Fallback to computing groups
        return ConsensusOutput._get_vote_groups(round_data.responses)

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
            vote_hash: str = ConsensusOutput._get_vote_hash(response)
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
            max_fields: Maximum fields to show in VERBOSE mode

        Returns:
            Formatted response summary
        """
        content_dict: Dict[str, Any] = response.content.model_dump()

        # Remove reasoning field as it's usually too long
        content_dict.pop("reasoning", None)

        if verbosity == VerbosityLevel.VERBOSE:
            # Show all fields
            return ConsensusOutput._format_dict_compact(content_dict)
        else:
            # Show only first few fields
            limited_dict: Dict[str, Any] = dict(list(content_dict.items())[:max_fields])
            if len(content_dict) > max_fields:
                limited_dict["..."] = f"({len(content_dict) - max_fields} more fields)"
            return ConsensusOutput._format_dict_compact(limited_dict)

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
            elif isinstance(value, str) and len(value) > ConsensusOutput._FIELD_SUMMARY_LENGTH:
                value_str = f'"{value[: ConsensusOutput._FIELD_SUMMARY_LENGTH - 3]}..."'
            else:
                value_str = json.dumps(value)
            items.append(f"{key}: {value_str}")

        return "\n".join(items)

    @staticmethod
    def format_result(
        result: "ConsensusResult",
        query: str,
        verbosity: VerbosityLevel = VerbosityLevel.VERBOSE,
    ) -> str:
        """Format the complete consensus result.

        In VERBOSE mode: Returns empty string (no output)
        In VERBOSE mode: Shows everything - all rounds, responses, metrics, evolution

        Args:
            result: The ConsensusResult object containing all consensus data
            query: The original query that was processed
            verbosity: Level of detail to display (VERBOSE = nothing, VERBOSE = everything)

        Returns:
            Formatted string representation of the complete consensus result
        """
        # In VERBOSE mode, show nothing
        if verbosity != VerbosityLevel.VERBOSE:
            return ""

        # VERBOSE mode - show everything
        output = []

        # Header
        output.append("\n" + "🌟" + "=" * (ConsensusOutput._SEPARATOR_WIDTH - 2) + "🌟")
        output.append("                    📊 CONSENSUS RESULT SUMMARY 📊")
        output.append("🌟" + "=" * (ConsensusOutput._SEPARATOR_WIDTH - 2) + "🌟")

        # Query Section
        output.append("\n💭 ORIGINAL QUERY:")
        output.append("-" * ConsensusOutput._SEPARATOR_WIDTH)
        wrapped_query = ConsensusOutput._wrap_text(query, ConsensusOutput._SEPARATOR_WIDTH - 4)
        for line in wrapped_query.split("\n"):
            output.append(f"  {line}")

        # Consensus Status
        output.append("\n📈 CONSENSUS STATUS:")
        output.append("-" * ConsensusOutput._SEPARATOR_WIDTH)

        if result.consensus_achieved:
            output.append(f"  ✅ CONSENSUS ACHIEVED in {result.total_rounds} round(s)")
            output.append(f"  🎯 Convergence Score: {result.convergence_score:.2%}")
        else:
            output.append(f"  ⚠️  NO CONSENSUS after {result.total_rounds} round(s)")
            output.append("  📊 Final Decision: Majority Vote")
            output.append(f"  📉 Convergence Score: {result.convergence_score:.2%}")

        # Participating Models
        output.append(f"\n  🤖 Participating Models ({len(result.participating_models)}):")
        for model_id in result.participating_models:
            contribution = result.metrics.model_contributions.get(model_id, 0.0) if result.metrics else 0.0
            status = "✓" if model_id not in result.dissenting_models else "✗"
            output.append(f"    {status} {model_id} (contribution: {contribution:.2%})")

        # Dissenting Models (if any)
        if result.dissenting_models:
            output.append(f"\n  ⚠️  Dissenting Models ({len(result.dissenting_models)}):")
            for model_id in result.dissenting_models:
                output.append(f"    • {model_id}")

        # Performance Metrics
        output.append("\n⚡ PERFORMANCE METRICS:")
        output.append("-" * ConsensusOutput._SEPARATOR_WIDTH)

        metrics = result.metrics
        output.append(f"  ⏱️  Duration: {metrics.duration_ms:.1f}ms")
        output.append(f"  🔄 Total Rounds: {metrics.rounds_to_convergence}")
        output.append(f"  📞 Total Model Calls: {metrics.total_model_calls}")
        output.append(f"  🎯 Consensus Confidence: {metrics.consensus_confidence:.2%}")

        if metrics.dissent_rate > 0:
            output.append(f"  ⚠️  Dissent Rate: {metrics.dissent_rate:.2%}")

        output.append(f"  🔀 Total Refinements: {metrics.total_refinements}")
        output.append(f"  📊 Avg Refinements/Round: {metrics.avg_refinements_per_round:.1f}")

        # Detailed Round-by-Round Display
        output.append("\n🔄 DETAILED ROUND-BY-ROUND BREAKDOWN:")
        output.append("=" * ConsensusOutput._SEPARATOR_WIDTH)

        for round_num, round_data in enumerate(result.rounds):
            output.append(f"\n  📍 ROUND {round_num}")
            output.append("  " + "-" * (ConsensusOutput._SEPARATOR_WIDTH - 2))

            # Round metrics
            num_models = len(round_data.responses)
            unique_votes = len(ConsensusOutput._get_unique_votes(round_data.responses))
            # Calculate agreement with special handling for complete disagreement
            # - All models agree = 1.0
            # - All models disagree (each votes differently) = 0.0
            # - Partial agreement = largest_group / total_models
            #   e.g., 3 models where 2 agree = 2/3 = 0.667
            if num_models == 0:
                agreement = 0.0
            elif unique_votes == 1:
                agreement = 1.0  # All models agree
            elif unique_votes == num_models:
                agreement = 0.0  # All models completely disagree
            else:
                # Partial agreement: use largest group fraction
                vote_groups = ConsensusOutput._get_vote_groups_from_round(round_data)
                max_group_size = max(vote_groups.values()) if vote_groups else 1
                agreement = max_group_size / num_models

            output.append("    📊 Round Metrics:")
            output.append(f"      • Models: {num_models}")
            output.append(f"      • Unique Votes: {unique_votes}")
            output.append(f"      • Agreement: {agreement * 100:.1f}%")
            if round_data.consensus_achieved:
                output.append("      • Status: ✅ CONSENSUS ACHIEVED")
            else:
                output.append("      • Status: ⚠️  No consensus yet")

            # Model responses for this round
            output.append("\n    🤖 Model Responses:")

            # Group responses by vote
            vote_groups = ConsensusOutput._get_vote_groups_from_round(round_data)

            for response in round_data.responses:
                vote_hash = ConsensusOutput._get_response_vote_key(response, round_data)
                group_size = vote_groups.get(vote_hash, 1)
                consensus_status = ConsensusOutput._format_consensus_status_plain(group_size, num_models)

                output.append(f"\n      [{response.id}]")
                output.append(f"        Vote Hash: {vote_hash[: ConsensusOutput._VOTE_HASH_DISPLAY_LENGTH]}")
                output.append(f"        Status: {consensus_status}")

                # Show response content
                content_dict = response.content.model_dump()
                reasoning = content_dict.pop("reasoning", None)

                output.append("        Response:")
                for key, value in content_dict.items():
                    value_str = ConsensusOutput._format_value(value, VerbosityLevel.VERBOSE)
                    # Wrap long values
                    if len(value_str) > 60:
                        output.append(f"          {key}:")
                        wrapped = ConsensusOutput._wrap_text(value_str, 70)
                        for line in wrapped.split("\n"):
                            output.append(f"            {line}")
                    else:
                        output.append(f"          {key}: {value_str}")

                if reasoning:
                    output.append("        Reasoning:")
                    wrapped_reasoning = ConsensusOutput._wrap_text(reasoning, 70)
                    for line in wrapped_reasoning.split("\n"):
                        output.append(f"          {line}")

            # Disagreement analysis for this round
            if round_data.disagreement_analysis and round_data.disagreement_analysis.disagreement_fields:
                output.append("\n    ⚠️  Disagreements:")
                for (
                    field,
                    values,
                ) in round_data.disagreement_analysis.disagreement_fields.items():
                    unique_values = len(set(values))
                    output.append(f"      • {field}: {unique_values} unique values")
                    # Show the actual values
                    value_counts: Dict[str, int] = {}
                    for v in values:
                        value_counts[v] = value_counts.get(v, 0) + 1
                    for value, count in sorted(value_counts.items(), key=lambda x: x[1], reverse=True):
                        value_display = value if len(value) <= 50 else value[:47] + "..."
                        output.append(f'        - "{value_display}" ({count} votes)')

            # Information flow for this round
            if round_data.information_flow:
                output.append("\n    🔀 Information Flow:")
                for model, peers in round_data.information_flow.items():
                    output.append(f"      • {model} ← {', '.join(peers)}")

            # Response evolution for this round
            if round_data.response_evolutions:
                output.append("\n    📈 Response Evolution:")
                for evolution in round_data.response_evolutions:
                    if evolution.vote_changed:
                        output.append(f"      • {evolution.id}: CHANGED VOTE")
                        if evolution.influenced_by:
                            output.append(f"        Influenced by: {', '.join(evolution.influenced_by)}")
                    else:
                        output.append(f"      • {evolution.id}: maintained position")

        # Consensus Evolution Summary Table
        output.append("\n📊 CONSENSUS EVOLUTION SUMMARY:")
        output.append("-" * ConsensusOutput._SEPARATOR_WIDTH)

        # Create evolution table
        header_format = "{:<8} {:<8} {:<15} {:<12} {:<25}"
        output.append(
            header_format.format(
                "Round",
                "Models",
                "Unique Votes",
                "Agreement %",
                "Key Changes",
            )
        )
        output.append("-" * ConsensusOutput._SEPARATOR_WIDTH)

        for i, round_data in enumerate(result.rounds):
            num_models = len(round_data.responses)
            unique_votes = len(ConsensusOutput._get_unique_votes(round_data.responses))
            # Calculate agreement with special handling for complete disagreement
            # - All models agree = 1.0
            # - All models disagree (each votes differently) = 0.0
            # - Partial agreement = largest_group / total_models
            #   e.g., 3 models where 2 agree = 2/3 = 0.667
            if num_models == 0:
                agreement = 0.0
            elif unique_votes == 1:
                agreement = 1.0  # All models agree
            elif unique_votes == num_models:
                agreement = 0.0  # All models completely disagree
            else:
                # Partial agreement: use largest group fraction
                vote_groups = ConsensusOutput._get_vote_groups_from_round(round_data)
                max_group_size = max(vote_groups.values()) if vote_groups else 1
                agreement = max_group_size / num_models

            # Determine key changes
            key_changes = []
            if i == 0:
                key_changes.append("Initial responses")
            else:
                if round_data.response_evolutions:
                    vote_changes = sum(1 for e in round_data.response_evolutions if e.vote_changed)
                    if vote_changes > 0:
                        key_changes.append(f"{vote_changes} vote changes")
                if round_data.consensus_achieved:
                    key_changes.append("✓ Consensus!")

            row = header_format.format(
                str(i),
                str(num_models),
                str(unique_votes),
                f"{agreement * 100:.1f}%",
                ", ".join(key_changes) if key_changes else "—",
            )
            output.append(row)

        # Final Response
        output.append("\n🎯 FINAL CONSENSUS RESPONSE:")
        output.append("-" * ConsensusOutput._SEPARATOR_WIDTH)

        # Format the final response
        response_dict = result.final_response.model_dump()

        # Show all fields in verbose mode
        for key, value in response_dict.items():
            value_str = ConsensusOutput._format_value(value, VerbosityLevel.VERBOSE)
            if key == "reasoning":
                output.append(f"\n  {key}:")
                wrapped_reasoning = ConsensusOutput._wrap_text(
                    value_str.strip('"'), ConsensusOutput._SEPARATOR_WIDTH - 6
                )
                for line in wrapped_reasoning.split("\n"):
                    output.append(f"    {line}")
            else:
                output.append(f"  {key}: {value_str}")

        # Reasoning Summary
        if result.reasoning:
            output.append("\n💡 CONSENSUS PROCESS SUMMARY:")
            output.append("-" * ConsensusOutput._SEPARATOR_WIDTH)
            wrapped_reasoning = ConsensusOutput._wrap_text(result.reasoning, ConsensusOutput._SEPARATOR_WIDTH - 4)
            for line in wrapped_reasoning.split("\n"):
                output.append(f"  {line}")

        # Footer
        output.append("\n" + "🌟" + "=" * (ConsensusOutput._SEPARATOR_WIDTH - 2) + "🌟")
        output.append("                    ✨ END OF CONSENSUS RESULT ✨")
        output.append("🌟" + "=" * (ConsensusOutput._SEPARATOR_WIDTH - 2) + "🌟\n")

        return "\n".join(output)

    @staticmethod
    def _wrap_text(text: str, max_width: int) -> str:
        """Wrap text to fit within specified width.

        Args:
            text: Text to wrap
            max_width: Maximum line width

        Returns:
            Wrapped text
        """

        return textwrap.fill(text, width=max_width, break_long_words=False, break_on_hyphens=False)
