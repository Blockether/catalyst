"""
-Inspired Consensus Mechanism for Reliable Multi-Model Reasoning

This implementation provides a consensus mechanism for combining outputs from multiple
reasoning models (RMs) to reduce hallucinations and improve accuracy by treating models
like nodes in a distributed network, inspired by the Hashgraph consensus algorithm.

"""

import hashlib
import json
import logging
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import (
    Any,
    Coroutine,
    DefaultDict,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Type,
    TypeVar,
    Union,
    cast,
)

import anyio
from pydantic import BaseModel, Field, RootModel

from com_blockether_catalyst.encoder.EncoderCore import EncoderCore
from com_blockether_catalyst.utils.TypedCalls import ArityOneTypedCall

from .ConsensusTypes import (
    ConsensusMetrics,
    ConsensusResult,
    ConsensusRound,
    ConsensusSettings,
    DisagreementAnalysis,
    FieldChangeValue,
    GossipHistory,
    ModelConfiguration,
    ModelMetrics,
    ModelResponse,
    ResponseEvolution,
    ResponseMetadata,
    RoundMetric,
    VerbosityLevel,
)
from .VotingComparison import (
    BaseModelWithReasoning,
    ComparisonStrategy,
    FieldComparator,
)

# Type variable for structured outputs
T = TypeVar("T", bound=BaseModelWithReasoning)

logger = logging.getLogger(__name__)


class Consensus(
    Generic[T],
    ArityOneTypedCall[str, ConsensusResult[T]],
):
    """
    -inspired consensus mechanism for multi-model reasoning.

    This class implements a consensus algorithm using multiple LLMs as
    validators in a distributed consensus protocol.

    The consensus mechanism works with any ArityOneTypedCall implementation,
    allowing for structured outputs from various LLM providers.

    Features:
    - Model contribution tracking for understanding consensus quality
    - Token usage and cost tracking
    - Direct voting comparison of structured responses
    - Enhanced metrics with confidence intervals
    - Concurrent execution control for performance
    """

    def __init__(
        self,
        models: List[ModelConfiguration[T]],
        judge: ArityOneTypedCall[str, T],
        settings: Optional[ConsensusSettings] = None,
    ) -> None:
        """
        Initialize the consensus mechanism with majority voting.

        Args:
            models: Model configurations for consensus
            judge: REQUIRED judge TypedCall for tie-breaking that returns same type T
            settings: Optional configuration settings. Uses defaults if not provided.
        """
        # Validate and store consensus configuration
        if not models:
            raise ValueError("At least one model must be specified")
        if not judge:
            raise ValueError("Judge must be provided for tie-breaking")

        self._models = models
        self._judge = judge  # Judge for tie-breaking

        # Use provided settings or create defaults
        self._settings = settings or ConsensusSettings()

    @property
    def settings(self) -> ConsensusSettings:
        """Get the consensus settings."""
        return self._settings

    @property
    def models(self) -> List[ModelConfiguration[T]]:
        """Get the list of model configurations."""
        return self._models

    async def call(self, x: str) -> ConsensusResult[T]:
        """
        Execute the  consensus algorithm (ArityOneTypedCall implementation).

        Args:
            x: The query string to get consensus on

        Returns:
            ConsensusResult with the final consensus response and metrics
        """

        start_time = datetime.now(timezone.utc)
        rounds: List[ConsensusRound] = []

        # Validate models
        if not self._models:
            raise ValueError("At least one model must be specified")

        enabled_models = self._models
        if not enabled_models:
            raise ValueError("At least one model must be enabled")

        # Use stored parameters
        max_rounds = self._settings.max_rounds
        convergence_threshold = self._settings.threshold
        first_round_threshold = self._settings.first_round_threshold

        # Initial round - get independent responses
        initial_round = await self._execute_initial_round(x)
        rounds.append(initial_round)

        # Check for early consensus with higher threshold for first round
        if await self._check_consensus(initial_round, first_round_threshold):
            assert (
                initial_round.consensus_response is not None
            ), "Consensus response must be set when consensus is achieved"
            result = self._create_result(
                rounds,
                initial_round.consensus_response,
                start_time,
            )
            self._log_final_consensus_result(result, start_time)
            return result

        # Iterative refinement rounds (use normal threshold)
        for round_num in range(1, max_rounds):
            # Execute gossip round
            consensus_round = await self._execute_gossip_round(x, rounds, round_num)
            rounds.append(consensus_round)

            # Check for consensus with normal threshold for subsequent rounds
            if await self._check_consensus(consensus_round, convergence_threshold):
                assert (
                    consensus_round.consensus_response is not None
                ), "Consensus response must be set when consensus is achieved"
                result = self._create_result(
                    rounds,
                    consensus_round.consensus_response,
                    start_time,
                )
                self._log_final_consensus_result(result, start_time)
                return result

        # Fallback: Use judge when no models succeeded or no consensus after max rounds
        fallback_method = None
        if not rounds[-1].responses:
            # All models failed - use judge directly
            judge_prompt = f"All models failed to respond. Please provide your best answer to: {x}"
            final_response = await self._judge.call(judge_prompt)
            fallback_method = "judge_all_failed"
            # Mark that judge was used as fallback
            rounds[-1].disagreement_analysis = DisagreementAnalysis(
                disagreement_fields={"all_models": ["failed"]},
                consensus_fields=[],
            )
        else:
            # Use majority vote - it handles tie detection internally
            response_type = type(rounds[-1].responses[0].content)
            final_response = await self._majority_vote(rounds[-1], response_type)

            # Determine if judge was used for tie-breaking by checking the stored vote groups
            if rounds[-1].vote_groups:
                vote_counts = rounds[-1].vote_groups
            else:
                # Fallback if vote_groups not set (shouldn't happen)
                vote_counts = {}
                for resp in rounds[-1].responses:
                    group_key = self._get_voting_group(resp.content)
                    if group_key not in vote_counts:
                        vote_counts[group_key] = []
                    vote_counts[group_key].append(resp)

            sorted_groups = sorted(vote_counts.items(), key=lambda x: len(x[1]), reverse=True)

            # Check if there was a tie
            if len(sorted_groups) > 1 and len(sorted_groups[0][1]) == len(sorted_groups[1][1]):
                fallback_method = "judge_tie"
            else:
                fallback_method = "majority_vote"

        result = self._create_result(rounds, final_response, start_time, fallback_method)
        self._log_final_consensus_result(result, start_time)

        return result

    async def _execute_initial_round(self, query: str) -> ConsensusRound[T]:
        """Execute the initial round where models respond independently."""
        responses = []

        # Create tasks for parallel execution with concurrency control
        tasks = []
        model_configs = self._models

        for model_config in model_configs:
            prompt = f"Perspective/Rules of model to provide valid response: {model_config.perspective}\n\nUser query:\n\n{query}"
            coro = self._get_model_response(model_config, prompt)
            tasks.append(coro)

        # Execute with controlled concurrency
        results = await self._execute_with_concurrency_limit(tasks)

        # Process results
        for model_config, result in zip(model_configs, results):
            if isinstance(result, Exception):
                logger.error(f"Model {model_config.id} failed: {result}")
                continue

            # At this point, result is guaranteed to be T (not Exception)
            # Cast is safe here because we checked for Exception above
            result_typed = cast(T, result)

            response = ModelResponse[T](
                id=model_config.id,
                round_number=0,
                content=result_typed,
                metadata=ResponseMetadata(initial_response=True),
                gossip_history=[
                    GossipHistory(
                        round_number=0,
                        refined_from_peers=False,
                        peer_models_seen=[],
                    )
                ],
            )
            responses.append(response)

        # Analyze initial disagreements
        disagreement_analysis = self._analyze_disagreements(responses)

        return ConsensusRound(
            round_number=0,
            responses=responses,
            information_flow={},
            disagreement_analysis=disagreement_analysis,
        )

    async def _execute_gossip_round(
        self,
        query: str,
        previous_rounds: List[ConsensusRound],
        round_num: int,
    ) -> ConsensusRound:
        """Execute a gossip round where models see each other's responses."""
        responses = []
        round_evolutions = []  # Initialize evolution tracking at the start
        previous_responses = previous_rounds[-1].responses

        # Create refinement tasks
        tasks = []
        model_configs = self._models

        for model_config in model_configs:
            # Get this model's previous response
            model_prev_response = next(
                (r for r in previous_responses if r.id == model_config.id),
                None,
            )

            if not model_prev_response:
                continue

            # Get peer responses
            peer_responses = [r for r in previous_responses if r.id != model_config.id]

            # Create refinement prompt with disagreement analysis
            refinement_prompt = self._create_refinement_prompt(
                query,
                model_prev_response,
                peer_responses,
                previous_round=previous_rounds[-1] if previous_rounds else None,
            )

            coro = self._get_model_response(model_config, refinement_prompt)
            tasks.append((model_config, coro, model_prev_response, peer_responses))

        # Execute all refinement tasks with controlled concurrency
        task_list = [t[1] for t in tasks]
        results = await self._execute_with_concurrency_limit(task_list)

        # Process results
        for (model_config, _, prev_response, peer_responses), result in zip(tasks, results):
            try:
                if isinstance(result, Exception):
                    logger.error(f"Model {model_config.id} refinement failed: {result}")
                    continue

                # Ensure result is BaseModel (not Exception) and cast to T
                assert isinstance(result, BaseModel)
                result_typed = cast(T, result)

                # Update gossip history
                new_history = prev_response.gossip_history.copy()
                new_history.append(
                    GossipHistory(
                        round_number=round_num,
                        refined_from_peers=True,
                        peer_models_seen=[p.id for p in peer_responses],
                    )
                )

                response = ModelResponse[T](
                    id=model_config.id,
                    round_number=round_num,
                    content=result_typed,
                    metadata=ResponseMetadata(refined=True, round=round_num),
                    gossip_history=new_history,
                )
                responses.append(response)

                # Track response evolution
                evolution = self._track_response_evolution(
                    prev_response,
                    response,
                    peer_responses,
                )
                round_evolutions.append(evolution)
            except Exception as e:
                logger.error(f"Model {model_config.id} processing failed: {e}")

        # Analyze disagreements
        disagreement_analysis = self._analyze_disagreements(responses)

        return ConsensusRound(
            round_number=round_num,
            responses=responses,
            information_flow=self._calculate_information_flow(responses),
            response_evolutions=round_evolutions,
            disagreement_analysis=disagreement_analysis,
        )

    async def _get_model_response(self, model_config: ModelConfiguration[T], prompt: str) -> T:
        """Get a structured response from a specific model."""
        try:
            response = await model_config.executor.call(prompt)
            return response
        except Exception as e:
            logger.error(f"Error calling model {model_config.id}: {e}")
            raise

    def _create_refinement_prompt(
        self,
        original_query: str,
        model_response: ModelResponse,
        peer_responses: List[ModelResponse],
        previous_round: Optional[ConsensusRound] = None,
    ) -> str:
        """Create an enhanced prompt for model refinement with disagreement analysis."""
        # Serialize structured responses to JSON for the prompt
        model_content = model_response.content.model_dump_json(indent=2)

        prompt = f"""Original Question: {original_query}

Your Previous Answer:
{model_content}

Peer Model Responses:
"""

        for i, peer in enumerate(peer_responses, 1):
            peer_content = peer.content.model_dump_json(indent=2)
            prompt += f"\nModel {i}:\n{peer_content}\n"

        # Add disagreement analysis if available from previous round
        if previous_round and previous_round.disagreement_analysis:
            analysis = previous_round.disagreement_analysis
            prompt += "\n## KEY INSIGHTS:\n"

            if analysis.consensus_fields:
                # Filter out any IGNORED fields from consensus display
                relevant_consensus = [
                    f for f in analysis.consensus_fields if f != "reasoning"
                ]  # reasoning is always ignored
                if relevant_consensus:
                    prompt += f"✓ Consensus reached on: {', '.join(relevant_consensus[:5])}\n"

            if analysis.disagreement_fields:
                # Only show non-ignored fields in disagreements
                # but we double-check here for safety
                disagreement_field_names = [f for f in analysis.disagreement_fields.keys() if f != "reasoning"][:3]
                if disagreement_field_names:
                    prompt += f"⚠ Fields with disagreement: {', '.join(disagreement_field_names)}\n"

        # Add consensus status concisely
        if previous_round:
            if previous_round.consensus_achieved:
                prompt += "✓ Near consensus - minor adjustments needed\n"
            else:
                prompt += "→ Consensus not yet reached - continue refinement\n"

        prompt += """
## REFINEMENT APPROACH:

**Benefits of Consensus:**
• Higher collective accuracy through diverse perspectives
• Reduced individual biases and blind spots
• Stronger confidence in final answer

**Example of Good Refinement:**
If you said "value: 100" but peers said "value: 150, 160, 155":
→ Good: "Adjusting to 155 based on peer calculations showing [specific evidence]"
→ Poor: "Keeping 100 despite all disagreement" (without exceptional proof)

**Your Task:**
1. **Convergence Incentive**: Answers closer to consensus receive higher weight
2. **Outlier Penalty**: Unjustified outliers reduce your influence
3. **Quick Scan**: Focus on fields where you differ significantly
4. **Evidence Required**: Only maintain outlier positions with concrete proof

**Critical Rule**: If 2+ models agree and you disagree, you MUST either:
- Adopt their position, OR
- Provide specific evidence why they're wrong

Your refined response (same JSON structure):"""

        return prompt

    async def _check_consensus(self, round_data: ConsensusRound[T], threshold: float) -> bool:
        """Check if consensus has been achieved using majority voting."""
        if len(round_data.responses) < 2:
            if round_data.responses:
                round_data.consensus_achieved = True
                round_data.consensus_response = round_data.responses[0].content
                return True
            # If no responses (all models failed), don't claim consensus yet
            return False

        responses = round_data.responses

        # Group responses for voting with fresh cache for this round
        vote_counts: Dict[str, List[ModelResponse[T]]] = {}
        voting_cache: Dict[int, str] = {}
        voting_groups: List[T] = []

        for response in responses:
            # Get or create voting group for this response
            response_group = self._get_voting_group(response.content, voting_cache, voting_groups)
            if response_group not in vote_counts:
                vote_counts[response_group] = []
            vote_counts[response_group].append(response)

        # Store the vote groups in the round for later use
        round_data.vote_groups = vote_counts

        # Check if we have a clear majority
        total_votes = len(responses)

        # Sort vote groups by count
        sorted_votes = sorted(vote_counts.items(), key=lambda x: len(x[1]), reverse=True)
        top_vote_count = len(sorted_votes[0][1])

        # Check if we need the judge for a tie
        if len(sorted_votes) > 1 and len(sorted_votes[0][1]) == len(sorted_votes[1][1]):
            # We have a tie - consensus not achieved yet, will continue rounds
            return False

        # Check if the top vote meets the threshold requirement
        required_votes = threshold * total_votes

        # Check for consensus based on threshold
        if top_vote_count >= required_votes:
            # Consensus achieved if threshold is met
            winner = sorted_votes[0][1][0]
            round_data.consensus_achieved = True
            round_data.consensus_response = winner.content
            return True

        return False

    def _calculate_information_flow(self, responses: List[ModelResponse]) -> Dict[str, List[str]]:
        """Calculate which models influenced each other in this round."""
        flow: Dict[str, List[str]] = {}

        for response in responses:
            if response.gossip_history:
                latest_history = response.gossip_history[-1]
                if latest_history.refined_from_peers and latest_history.peer_models_seen:
                    # This model was influenced by its peers
                    flow[response.id] = latest_history.peer_models_seen

        return flow

    async def _majority_vote(self, round_data: ConsensusRound[T], response_type: Type[T]) -> T:
        """Perform majority voting on structured responses with judge for tie-breaking."""
        if not round_data.responses:
            raise ValueError("No responses for majority vote")

        responses = round_data.responses

        # Use stored vote groups if available, otherwise compute them
        if round_data.vote_groups:
            vote_counts = round_data.vote_groups
        else:
            # Group responses for voting with fresh cache
            vote_counts: Dict[str, List[ModelResponse[T]]] = {}
            voting_cache: Dict[int, str] = {}
            voting_groups: List[T] = []

            for response in responses:
                response_group = self._get_voting_group(response.content, voting_cache, voting_groups)
                if response_group not in vote_counts:
                    vote_counts[response_group] = []
                vote_counts[response_group].append(response)

            # Store the computed groups
            round_data.vote_groups = vote_counts

        # Sort by vote count
        sorted_votes = sorted(vote_counts.items(), key=lambda x: len(x[1]), reverse=True)

        # Check for tie at the top
        if len(sorted_votes) > 1 and len(sorted_votes[0][1]) == len(sorted_votes[1][1]):
            # We have a tie - collect ALL tied responses
            top_vote_count = len(sorted_votes[0][1])
            tied_responses = []
            for group_key, responses in sorted_votes:
                if len(responses) == top_vote_count:
                    # Add the first response from each tied group
                    tied_responses.append(responses[0])
                else:
                    # No more ties, stop collecting
                    break

            winner = await self._invoke_judge_for_tiebreak(tied_responses, round_data)
            return winner.content

        # Clear winner - return the response with most votes
        return sorted_votes[0][1][0].content

    async def _invoke_judge_for_tiebreak(
        self, tied_responses: List[ModelResponse[T]], round_data: ConsensusRound[T]
    ) -> ModelResponse[T]:
        """Invoke the judge to break a tie between responses.

        The judge analyzes the tied responses and the voting history (gossips)
        to synthesize the best response based on the quality of reasoning.
        """
        # Prepare judge prompt with voting history and ALL tied responses
        judge_prompt = f"""As a neutral judge, you must break the tie between {len(tied_responses)} responses and provide the best synthesis.

## Voting History:
Round {round_data.round_number} had {len(round_data.responses)} total votes.
There is a {len(tied_responses)}-way tie at the top.

## Tied Responses:

"""
        # Add all tied responses to the prompt
        for i, response in enumerate(tied_responses, 1):
            judge_prompt += f"""Response {i} (from {response.id}):
{response.content.model_dump_json(indent=2)}

"""

        judge_prompt += """## Your Task:
Analyze all tied responses and provide YOUR OWN response that either:
1. Selects the best response from those provided
2. Synthesizes the best elements from multiple responses
3. Provides an improved answer based on their collective insights

Base your decision on:
- Quality of reasoning
- Internal consistency
- Supporting evidence
- Completeness of answer

Provide a response in the same JSON format as the tied responses above.
"""

        # Call the judge - it returns type T directly
        judge_response = await self._judge.call(judge_prompt)

        return ModelResponse[T](
            id="judge",
            round_number=round_data.round_number,
            content=judge_response,
            metadata=ResponseMetadata(judge_decision=True, resolved_tie=True),
            gossip_history=[],
        )

    def _create_result(
        self,
        rounds: List[ConsensusRound[T]],
        final_response: T,
        start_time: datetime,
        fallback_method: Optional[str] = None,
    ) -> ConsensusResult[T]:
        """Create the final consensus result."""
        end_time = datetime.now(timezone.utc)
        duration_ms = (end_time - start_time).total_seconds() * 1000

        # Identify dissenting models
        final_round = rounds[-1]
        dissenting_models = []

        # Only identify dissenting models if consensus was NOT achieved
        # When consensus is achieved, all models contributed to it even if the
        # final consensus response doesn't exactly match any individual response
        if not final_round.consensus_achieved:
            # No consensus - use voting patterns to find outliers
            responses = [r.content for r in final_round.responses]
            if len(responses) > 1:
                # Find responses that are significantly different from others
                for i, response in enumerate(final_round.responses):
                    is_outlier = self._is_response_outlier(response, final_round.responses)
                    if is_outlier:
                        dissenting_models.append(response.id)

        # Calculate information flow metrics
        total_refinements = sum(
            1 if r.gossip_history and r.gossip_history[-1].refined_from_peers else 0
            for round in rounds
            for r in round.responses
        )

        # Calculate model contribution analysis
        model_contributions = self._calculate_model_contributions(rounds)

        # Calculate quality metrics
        consensus_confidence = self._calculate_consensus_confidence(
            final_round.consensus_achieved,
            len(rounds),
            len(dissenting_models),
            len(self._models),
            rounds,
        )

        # Create strongly typed metrics
        metrics = ConsensusMetrics(
            duration_ms=duration_ms,
            rounds_to_convergence=len(rounds),
            total_model_calls=sum(len(r.responses) for r in rounds),
            convergence_achieved=final_round.consensus_achieved,
            dissent_rate=(len(dissenting_models) / len(self._models) if self._models else 0),
            model_contributions=model_contributions,
            consensus_confidence=consensus_confidence,
            convergence_indicator=self._calculate_convergence_score(rounds),
            total_refinements=total_refinements,
            avg_refinements_per_round=total_refinements / len(rounds) if rounds else 0,
            information_flows=[round.information_flow for round in rounds],
            fallback_method=fallback_method,
        )

        # Generate reasoning based on the consensus process
        reasoning = self._generate_consensus_reasoning(
            rounds, final_round.consensus_achieved, dissenting_models, fallback_method
        )

        return ConsensusResult(
            reasoning=reasoning,
            consensus_achieved=final_round.consensus_achieved,
            final_response=final_response,
            rounds=rounds,
            total_rounds=len(rounds),
            convergence_score=self._calculate_convergence_score(rounds),
            participating_models=[m.id for m in self._models],
            dissenting_models=dissenting_models,
            model_contributions=model_contributions,
            metrics=metrics,
        )

    def _generate_consensus_reasoning(
        self,
        rounds: List[ConsensusRound[T]],
        consensus_achieved: bool,
        dissenting_models: List[str],
        fallback_method: Optional[str] = None,
    ) -> str:
        """Generate reasoning explanation for the consensus result."""
        if consensus_achieved:
            return f"Consensus was successfully achieved after {len(rounds)} round(s) of deliberation. All participating models converged to agreement through iterative refinement and peer collaboration. The final response represents the collective wisdom of all {len(self._models)} models."
        else:
            dissent_info = (
                f" with {len(dissenting_models)} dissenting model(s): {', '.join(dissenting_models)}"
                if dissenting_models
                else ""
            )
            # Use the provided fallback method
            if fallback_method == "judge_all_failed":
                method_desc = "judge decision (all models failed)"
            elif fallback_method == "judge_tie":
                method_desc = "judge decision (tie detected)"
            elif fallback_method == "majority_vote":
                method_desc = "majority voting"
            else:
                method_desc = "fallback mechanism"

            return f"Consensus was not achieved after {len(rounds)} round(s) of deliberation{dissent_info}. The system fell back to {method_desc} to determine the final response. Despite the lack of full agreement, this represents the best collective judgment available."

    def _calculate_convergence_score(self, rounds: List[ConsensusRound[T]]) -> float:
        """Calculate overall convergence score across all rounds using voting patterns."""
        if not rounds:
            return 0.0

        final_round = rounds[-1]
        if final_round.consensus_achieved:
            return 1.0

        # Calculate voting convergence across rounds
        convergence_scores = []

        # Track voting agreement improvement over rounds
        for i in range(len(rounds)):
            round_agreement = self._calculate_round_similarity(rounds[i].responses)
            convergence_scores.append(round_agreement)

        if not convergence_scores:
            return 0.0

        # Calculate convergence as a combination of:
        # 1. Final round agreement (60% weight)
        # 2. Improvement trend (40% weight)
        final_agreement = convergence_scores[-1]

        # Calculate improvement trend if we have multiple rounds
        improvement = 0.0
        if len(convergence_scores) > 1:
            # Check if agreement is increasing
            improvements = []
            for i in range(1, len(convergence_scores)):
                improvement_i = convergence_scores[i] - convergence_scores[i - 1]
                improvements.append(max(0, improvement_i))  # Only count positive improvements

            if improvements:
                improvement = sum(improvements) / len(improvements)

        # Combine final agreement with improvement trend
        return (final_agreement * 0.6) + (improvement * 0.4)

    def _calculate_round_similarity(self, responses: List[ModelResponse]) -> float:
        """Calculate voting agreement ratio - how many models voted the same."""
        if len(responses) < 2:
            return 1.0

        # Count unique vote groups
        vote_groups = set()
        for response in responses:
            response_group = self._get_voting_group(response.content)
            vote_groups.add(response_group)

        # Agreement ratio: 1.0 if all same, approaching 0 if all different
        agreement_ratio = 1.0 - ((len(vote_groups) - 1) / (len(responses) - 1))
        return agreement_ratio

    # Model contribution analysis methods
    def _calculate_model_contributions(self, rounds: List[ConsensusRound[T]]) -> Dict[str, float]:
        """
        Calculate how much each model contributed to achieving consensus.

        This measures positive contribution through:
        - Consistency in responses
        - Convergence toward final consensus
        - Quality of reasoning provided
        """
        model_contributions = {}

        # Analyze each model's behavior across rounds
        model_responses: DefaultDict[str, List[ModelResponse]] = defaultdict(list)

        for round_data in rounds:
            for response in round_data.responses:
                model_responses[response.id].append(response)

        # Calculate contribution metrics for each model
        for model_id, responses in model_responses.items():
            # Calculate contribution score directly without storing metrics
            contribution_score = self._calculate_contribution_score(responses, rounds)
            model_contributions[model_id] = contribution_score

        return model_contributions

    def _calculate_contribution_score(
        self,
        model_responses: List[ModelResponse],
        all_rounds: List[ConsensusRound[T]],
    ) -> float:
        """Calculate how much a model contributed to consensus quality."""
        if not model_responses:
            return 0.0

        # Factor 1: Consistency (50%) - stable reasoning across rounds
        consistency_score = self._calculate_consistency_score(model_responses)

        # Factor 2: Convergence (50%) - moves toward final consensus
        convergence_score = self._calculate_convergence_contribution(model_responses, all_rounds)

        return 0.5 * consistency_score + 0.5 * convergence_score

    def _calculate_convergence_contribution(
        self,
        model_responses: List[ModelResponse],
        all_rounds: List[ConsensusRound[T]],
    ) -> float:
        """Measure how much the model helped achieve convergence through voting."""
        if len(model_responses) < 2:
            return 1.0

        # Get the final consensus response if available
        final_consensus = None
        for round_data in reversed(all_rounds):
            if round_data.consensus_achieved and round_data.consensus_response:
                final_consensus = round_data.consensus_response
                break

        if not final_consensus:
            # No consensus achieved, measure consistency as proxy
            return self._calculate_consistency_score(model_responses)

        # Check if model voted for the winning consensus
        final_group = self._get_voting_group(final_consensus)
        voted_correctly = 0

        for response in model_responses:
            response_group = self._get_voting_group(response.content)
            if response_group == final_group:
                voted_correctly += 1

        # Score based on how often model voted for consensus
        base_score = voted_correctly / len(model_responses)

        # Bonus if model converged to consensus over time
        if len(model_responses) >= 2:
            early_match = self._get_voting_group(model_responses[0].content) == final_group
            late_match = self._get_voting_group(model_responses[-1].content) == final_group
            if late_match and not early_match:
                return min(1.0, base_score + 0.3)

        return base_score

    def _is_response_outlier(
        self,
        response: ModelResponse[T],
        all_responses: List[ModelResponse[T]],
        threshold: float = 0.3,
    ) -> bool:
        """Check if a response is an outlier by voting pattern."""
        if len(all_responses) < 3:  # Need at least 3 for outlier detection
            return False

        # Count vote groups
        vote_counts: Dict[str, int] = {}
        for r in all_responses:
            response_group = self._get_voting_group(r.content)
            vote_counts[response_group] = vote_counts.get(response_group, 0) + 1

        # Get this response's vote group size
        response_group = self._get_voting_group(response.content)
        response_vote_count = vote_counts.get(response_group, 0)

        # Response is outlier if it's alone while others agree
        max_vote_count = max(vote_counts.values())
        if response_vote_count == 1 and max_vote_count >= len(all_responses) * (1 - threshold):
            return True

        return False

    def _calculate_consistency_score(self, responses: List[ModelResponse]) -> float:
        """Calculate consistency score for a model across rounds using vote hashes."""
        if len(responses) < 2:
            return 1.0  # Single response is perfectly consistent

        # Check if model maintains consistent voting
        vote_groups = []
        for response in responses:
            vote_groups.append(self._get_voting_group(response.content))

        # Count how many times the vote stayed the same between rounds
        consistent_votes = 0
        for i in range(len(vote_groups) - 1):
            if vote_groups[i] == vote_groups[i + 1]:
                consistent_votes += 1

        return consistent_votes / (len(vote_groups) - 1) if len(vote_groups) > 1 else 1.0

    # Enhanced metrics methods
    def _calculate_consensus_confidence(
        self,
        consensus_achieved: bool,
        total_rounds: int,
        dissenting_count: int,
        total_models: int,
        rounds: List[ConsensusRound[T]],
    ) -> float:
        """Calculate voting strength as confidence metric."""
        if not consensus_achieved:
            return 0.0

        # For voting: confidence is based on vote proportion
        # If all models agree: 1.0
        # If bare majority: lower confidence
        voting_proportion = (total_models - dissenting_count) / max(total_models, 1)

        # Field-level consistency: how many fields reached consensus
        field_consistency = self._calculate_field_consistency(rounds)

        # Vote stability: how stable were votes across rounds
        vote_stability = self._calculate_vote_stability(rounds)

        # Combine metrics (weighted average)
        # 50% voting proportion, 30% field consistency, 20% vote stability
        agreement_strength = voting_proportion * 0.5 + field_consistency * 0.3 + vote_stability * 0.2

        return min(1.0, agreement_strength)

    def _calculate_field_consistency(self, round_infos: List[ConsensusRound]) -> float:
        """Calculate field-level consistency across voting groups.

        Returns a score from 0 to 1 indicating what proportion of fields
        reached consensus (not just the overall response).
        """
        if not round_infos:
            return 0.0

        # Get the latest round for analysis
        latest_round = round_infos[-1]

        # Track field consensus across voting groups
        field_consensus_count = 0
        total_fields = 0

        # Get all field differences from the round's disagreement analysis
        if latest_round.disagreement_analysis is not None:
            disagreement_fields = latest_round.disagreement_analysis.disagreement_fields
            consensus_fields = latest_round.disagreement_analysis.consensus_fields

            # Total fields is disagreement fields + consensus fields
            total_fields = len(disagreement_fields) + len(consensus_fields)
            field_consensus_count = len(consensus_fields)
        else:
            # If no field tracking, use overall consensus as proxy
            return 1.0 if latest_round.consensus_achieved else 0.0

        if total_fields == 0:
            return 1.0  # No fields to compare = perfect consistency

        return field_consensus_count / total_fields

    def _calculate_vote_stability(self, round_infos: List[ConsensusRound]) -> float:
        """Calculate vote stability across rounds.

        Returns a score from 0 to 1 indicating how stable votes were.
        Models that stick to their votes show stronger conviction.
        """
        if len(round_infos) <= 1:
            return 1.0  # Single round = perfectly stable

        # Track how models' responses group together across rounds
        vote_changes = 0
        total_comparisons = 0

        # Compare responses between consecutive rounds
        for i in range(1, len(round_infos)):
            prev_round = round_infos[i - 1]
            curr_round = round_infos[i]

            # Build voting groups for each round
            prev_groups = {}
            for response in prev_round.responses:
                group_key = self._get_voting_group(response.content)
                if group_key not in prev_groups:
                    prev_groups[group_key] = []
                prev_groups[group_key].append(response.id)

            curr_groups = {}
            for response in curr_round.responses:
                group_key = self._get_voting_group(response.content)
                if group_key not in curr_groups:
                    curr_groups[group_key] = []
                curr_groups[group_key].append(response.id)

            # Track which models changed their votes
            for model_config in self._models:
                model_id = model_config.id

                # Find model's vote in previous round
                prev_vote = None
                for group_id, members in prev_groups.items():
                    if model_id in members:
                        prev_vote = group_id
                        break

                # Find model's vote in current round
                curr_vote = None
                for group_id, members in curr_groups.items():
                    if model_id in members:
                        curr_vote = group_id
                        break

                # Check if vote changed
                if prev_vote is not None and curr_vote is not None:
                    total_comparisons += 1
                    if prev_vote != curr_vote:
                        vote_changes += 1

        if total_comparisons == 0:
            return 1.0  # No comparisons = perfect stability

        stability = 1.0 - (vote_changes / total_comparisons)
        return max(0.0, stability)

    # Voting Group Methods
    def _get_voting_group(
        self,
        response: T,
        voting_group_cache: Optional[Dict[int, str]] = None,
        voting_groups: Optional[List[T]] = None,
    ) -> str:
        """Get or create a voting group ID for this response.

        Groups similar responses together using field-specific comparison strategies.
        Returns a group ID like 'group_1', 'group_2', etc.

        Args:
            response: The response to group
            voting_group_cache: Optional cache for this round's voting groups
            voting_groups: Optional list of group representatives for this round

        If cache/groups not provided, creates temporary ones for comparison only.
        """
        # Use provided cache or create temporary one
        if voting_group_cache is None:
            voting_group_cache = {}
        if voting_groups is None:
            voting_groups = []

        # Check if we've seen this exact response object before
        response_id = id(response)
        if response_id in voting_group_cache:
            return voting_group_cache[response_id]

        # Find which group this response belongs to by comparing with representatives
        for i, group_representative in enumerate(voting_groups):
            if self._responses_are_similar(response, group_representative):
                # Found a matching group
                group_key = f"group_{i + 1}"
                voting_group_cache[response_id] = group_key
                return group_key

        # No matching group - create a new one with this response as representative
        voting_groups.append(response)
        group_key = f"group_{len(voting_groups)}"
        voting_group_cache[response_id] = group_key
        return group_key

    def _responses_are_similar(self, resp1: T, resp2: T) -> bool:
        """Check if two responses are similar enough to vote together.

        This compares each field according to its voting strategy.
        """
        # Get the model class and fields
        model_class = resp1.__class__
        model_fields = model_class.model_fields

        # Compare each field according to its strategy
        for field_name, field_info in model_fields.items():
            # Get field values
            value1 = getattr(resp1, field_name)
            value2 = getattr(resp2, field_name)

            # Get comparison strategy
            from .VotingComparison import ComparisonStrategy, FieldComparator

            voting_meta = FieldComparator._extract_voting_metadata(field_info)

            # Skip ignored fields
            if voting_meta.strategy == ComparisonStrategy.IGNORE:
                continue

            # Compare based on strategy
            if voting_meta.strategy == ComparisonStrategy.SEMANTIC:
                # For semantic comparison, check similarity
                if isinstance(value1, str) and isinstance(value2, str):
                    from ..encoder.EncoderCore import EncoderCore

                    emb1 = EncoderCore.encode_single(value1)
                    emb2 = EncoderCore.encode_single(value2)
                    similarity = EncoderCore.cosine_similarity(emb1, emb2)

                    if similarity < voting_meta.threshold:
                        return False  # Not similar enough
                else:
                    # Non-string semantic comparison - just check equality
                    if value1 != value2:
                        return False

            elif voting_meta.strategy == ComparisonStrategy.RANGE:
                # For range comparison, check if within tolerance
                if isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
                    if value1 == 0 and value2 == 0:
                        continue  # Both zero, consider equal
                    # Check relative difference
                    max_val = max(abs(value1), abs(value2))
                    if max_val > 0:
                        rel_diff = abs(value1 - value2) / max_val
                        if rel_diff > voting_meta.tolerance:
                            return False
                else:
                    if value1 != value2:
                        return False

            elif voting_meta.strategy == ComparisonStrategy.DERIVED:
                # For nested models, recursively compare
                if isinstance(value1, (BaseModel, RootModel)) and isinstance(value2, (BaseModel, RootModel)):
                    # Create temporary wrapper to use our comparison
                    if not self._responses_are_similar(cast(T, value1), cast(T, value2)):
                        return False
                else:
                    if value1 != value2:
                        return False

            else:
                # EXACT comparison (default)
                if value1 != value2:
                    return False

        # All non-ignored fields match according to their strategies
        return True

    def _track_response_evolution(
        self,
        prev_response: ModelResponse[T],
        new_response: ModelResponse[T],
        peer_responses: List[ModelResponse[T]],
    ) -> ResponseEvolution[T]:
        """Track how a response evolved between rounds."""
        prev_dict = prev_response.content.model_dump()
        new_dict = new_response.content.model_dump()

        # Find field changes
        field_changes: Dict[str, FieldChangeValue] = {}
        for key in prev_dict:
            if key in new_dict and prev_dict[key] != new_dict[key]:
                field_changes[key] = FieldChangeValue(old_value=prev_dict[key], new_value=new_dict[key])

        # Get voting groups for comparison
        prev_group = self._get_voting_group(prev_response.content)
        new_group = self._get_voting_group(new_response.content)

        # Check if vote changed
        vote_changed = prev_group != new_group

        # Extract reasoning evolution
        reasoning_evolution = ""
        if "reasoning" in prev_dict and "reasoning" in new_dict:
            if prev_dict["reasoning"] != new_dict["reasoning"]:
                reasoning_evolution = (
                    f"Changed from: {prev_dict['reasoning'][:100]}... to: {new_dict['reasoning'][:100]}..."
                )

        # Determine which models influenced this evolution
        influenced_by = []
        # Check if model changed vote to match any peer (using groups calculated above)

        if prev_group != new_group:  # Vote changed
            for peer in peer_responses:
                peer_group = self._get_voting_group(peer.content)
                if new_group == peer_group:
                    # Model adopted peer's position
                    influenced_by.append(peer.id)

        return ResponseEvolution[T](
            id=new_response.id,
            round_from=prev_response.round_number,
            round_to=new_response.round_number,
            field_changes=field_changes,
            vote_changed=vote_changed,
            reasoning_evolution=reasoning_evolution,
            influenced_by=influenced_by,
        )

    def _analyze_disagreements(
        self,
        responses: List[ModelResponse[T]],
    ) -> DisagreementAnalysis:
        """Analyze disagreements between responses with detailed comparison."""
        if not responses:
            return DisagreementAnalysis()

        analysis = DisagreementAnalysis()

        # Get field information from the first response's model class
        if responses and responses[0].content:
            model_class = responses[0].content.__class__
            model_fields = model_class.model_fields
        else:
            # If no responses, return empty analysis
            return analysis

        # Collect all field values with model IDs for traceability
        field_values: DefaultDict[str, List[str]] = defaultdict(list)
        field_values_with_models: DefaultDict[str, List[tuple[str, str]]] = defaultdict(list)

        for response in responses:
            response_dict = response.content.model_dump()
            for field, value in response_dict.items():
                str_value = str(value)
                field_values[field].append(str_value)
                field_values_with_models[field].append((response.id, str_value))

        # Track ignored fields for logging
        ignored_fields = []

        # Analyze each field
        for field, values in field_values.items():
            # Check if field should be ignored based on ComparisonStrategy
            field_info = model_fields.get(field)
            if field_info:
                # Extract voting metadata using the helper method
                voting_meta = FieldComparator._extract_voting_metadata(field_info)
                if voting_meta.strategy == ComparisonStrategy.IGNORE:
                    # Skip this field in disagreement analysis
                    ignored_fields.append(field)
                    if self._settings.verbosity == VerbosityLevel.VERBOSE:
                        logger.debug(f"    ⊘ Field '{field}' is IGNORED for consensus (strategy=IGNORE)")
                    continue

            unique_values = list(set(values))

            if len(unique_values) == 1:
                # Consensus on this field
                analysis.consensus_fields.append(field)
            else:
                # Disagreement on this field (but only for fields that matter)
                analysis.disagreement_fields[field] = values

                # Don't log during analysis - only show in final consolidated output
                pass

        return analysis

    def _wrap_text_for_logging(self, text: str, max_length: int = 80, indent: int = 0) -> str:
        """Wrap text for readable logging output.

        Args:
            text: Text to wrap
            max_length: Maximum line length
            indent: Number of spaces to indent wrapped lines

        Returns:
            Wrapped text with proper indentation
        """
        import textwrap

        # First line doesn't need indent, subsequent lines do
        wrapper = textwrap.TextWrapper(
            width=max_length,
            subsequent_indent=" " * indent,
            break_long_words=False,
            break_on_hyphens=False,
        )

        # Handle multiline text
        lines = text.split("\n")
        wrapped_lines = []
        for line in lines:
            if len(line) <= max_length:
                wrapped_lines.append(line)
            else:
                wrapped_lines.extend(wrapper.wrap(line))

        return "\n".join(wrapped_lines)

    def _log_field_analysis(self, field: str, unique_values: Dict[str, List[str]], field_info: Any = None) -> None:
        """Log appropriate analysis for a field based on its type and comparison strategy."""
        from ..consensus.VotingComparison import ComparisonStrategy, FieldComparator

        # Get comparison strategy and field type
        comparison_strategy = ComparisonStrategy.EXACT  # default
        field_type = str  # default

        if field_info:
            voting_meta = FieldComparator._extract_voting_metadata(field_info)
            comparison_strategy = voting_meta.strategy

            # Get the field's annotation/type
            field_annotation = getattr(field_info, "annotation", None)
            if field_annotation:
                # Handle Optional types and extract the actual type
                origin = getattr(field_annotation, "__origin__", None)
                if origin:
                    args = getattr(field_annotation, "__args__", ())
                    if args:
                        field_type = args[0]  # First arg for Optional[T] is T
                else:
                    field_type = field_annotation

        unique_value_list = list(unique_values.keys())

        # Different analysis based on comparison strategy and field type
        if comparison_strategy == ComparisonStrategy.SEMANTIC and field_type is str:
            # Semantic similarity for string fields
            logger.info(f"      📝  Semantic Analysis for '{field}' (string field):")
            self._log_similarity_matrix(field, unique_value_list)

        elif field_type is bool:
            # Boolean analysis
            logger.info(f"      🔀  Boolean Analysis for '{field}':")
            true_models = []
            false_models = []
            for value, models in unique_values.items():
                if value.lower() == "true":
                    true_models.extend(models)
                elif value.lower() == "false":
                    false_models.extend(models)

            total = len(true_models) + len(false_models)
            if total > 0:
                true_pct = (len(true_models) / total) * 100
                false_pct = (len(false_models) / total) * 100

                if true_models:
                    true_names = ", ".join(true_models)
                    logger.info(f"        ✓ True: {len(true_models)} models ({true_pct:.1f}%) - {true_names}")
                if false_models:
                    false_names = ", ".join(false_models)
                    logger.info(f"        ✗ False: {len(false_models)} models ({false_pct:.1f}%) - {false_names}")

        elif field_type in (int, float) or self._is_numeric_field(unique_value_list):
            # Numeric analysis
            logger.info(
                f"      📊  Numeric Analysis for '{field}' ({field_type.__name__ if hasattr(field_type, '__name__') else 'numeric'}):"
            )
            numeric_values = []
            value_to_models = {}
            for value_str, models in unique_values.items():
                try:
                    num_val = float(value_str)
                    numeric_values.append(num_val)
                    value_to_models[num_val] = models
                except ValueError:
                    continue

            if numeric_values:
                min_val = min(numeric_values)
                max_val = max(numeric_values)
                avg_val = sum(numeric_values) / len(numeric_values)
                range_val = max_val - min_val

                logger.info(f"        📈  Range: {min_val} → {max_val} (Δ{range_val})")
                logger.info(f"        📊  Average: {avg_val:.2f}")
                logger.info("        🎯  Values with models:")
                for value in sorted(set(numeric_values)):
                    models_list = value_to_models.get(value, [])
                    models_str = ", ".join(models_list)
                    logger.info(
                        f"           • {value} ({len(models_list)} model{'s' if len(models_list) != 1 else ''}) - {models_str}"
                    )

    def _is_numeric_field(self, values: List[str]) -> bool:
        """Check if all values in the list are numeric."""
        try:
            for value in values:
                float(value)
            return True
        except ValueError:
            return False

    def _log_similarity_matrix(self, field: str, unique_values: List[str]) -> None:
        """Log a pairwise semantic similarity matrix for unique values."""
        try:
            import numpy as np

            # Get embeddings for all values
            embeddings = []
            for value in unique_values:
                truncated = value[:1000] if len(value) > 1000 else value
                embedding = EncoderCore.encode_single(truncated)
                embeddings.append(embedding)

            # Calculate pairwise similarity matrix
            n = len(embeddings)
            similarity_matrix = np.zeros((n, n))

            for i in range(n):
                for j in range(n):
                    if i == j:
                        similarity_matrix[i][j] = 1.0
                    else:
                        # Cosine similarity
                        similarity = np.dot(embeddings[i], embeddings[j]) / (
                            np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                        )
                        similarity_matrix[i][j] = similarity

            # Log the similarity matrix
            logger.info(f"      Semantic Similarity Matrix for '{field}':")

            # Create a visual representation
            for i, val1 in enumerate(unique_values):
                similarities = []
                for j, val2 in enumerate(unique_values):
                    if i != j:
                        sim = similarity_matrix[i][j]
                        # Create visual indicator
                        if sim >= 0.8:
                            indicator = "≈≈≈"  # Very similar
                        elif sim >= 0.6:
                            indicator = "≈≈ "  # Moderately similar
                        elif sim >= 0.4:
                            indicator = "≈  "  # Somewhat similar
                        else:
                            indicator = "≠  "  # Different

                        # Wrap the compared value for readability
                        val2_wrapped = self._wrap_text_for_logging(val2, max_length=50, indent=0)
                        val2_preview = val2_wrapped.split("\n")[0]  # First line only for preview
                        if len(val2) > 50:
                            val2_preview += "..."
                        similarities.append(f"{indicator} {sim:.2f} to: {val2_preview}")

                # Wrap the main value
                val1_wrapped = self._wrap_text_for_logging(val1, max_length=60, indent=8)
                logger.info("        Value:")
                for line in val1_wrapped.split("\n"):
                    logger.info(f"          {line}")
                logger.info("        Similarities:")
                for sim_str in similarities:
                    logger.info(f"          {sim_str}")

            # Check for semantic clusters
            self._identify_semantic_clusters(unique_values, similarity_matrix)

        except Exception as e:
            logger.debug(f"Could not compute similarity matrix: {e}")

    def _identify_semantic_clusters(self, unique_values: List[str], similarity_matrix: Any) -> None:
        """Identify and log semantic clusters in the values."""
        import numpy as np

        # Find clusters (values with similarity > 0.7)
        clusters = []
        used = set()

        for i in range(len(unique_values)):
            if i in used:
                continue

            cluster = [i]
            for j in range(i + 1, len(unique_values)):
                if j not in used and similarity_matrix[i][j] >= 0.7:
                    cluster.append(j)
                    used.add(j)

            if len(cluster) > 1:
                used.add(i)
                clusters.append(cluster)

        if clusters:
            logger.info("      🔗 Semantic Clusters Detected:")
            for cluster_idx, cluster in enumerate(clusters, 1):
                logger.info(f"        Cluster {cluster_idx}: {len(cluster)} semantically similar values")
                for idx in cluster:
                    val = unique_values[idx]
                    wrapped_val = self._wrap_text_for_logging(val, max_length=65, indent=12)
                    lines = wrapped_val.split("\n")
                    logger.info(f"          • {lines[0]}")
                    for line in lines[1:]:
                        logger.info(f"            {line}")
                avg_sim = np.mean([similarity_matrix[cluster[0]][j] for j in cluster[1:]])
                logger.info(f"          Average similarity: {avg_sim:.2%}")

    async def _execute_with_concurrency_limit(
        self,
        tasks: List[Coroutine],
    ) -> List[Union[BaseModel, Exception]]:
        """Execute tasks with concurrency limit.

        Args:
            tasks: List of coroutines to execute

        Returns:
            List of results (or exceptions) in the same order as input tasks
        """
        max_concurrent = self._settings.max_concurrent_calls
        semaphore = anyio.Semaphore(max_concurrent)

        async def run_with_semaphore(task: Coroutine) -> Union[BaseModel, Exception]:
            async with semaphore:
                try:
                    result = await task
                    return cast(Union[BaseModel, Exception], result)
                except Exception as e:
                    return e

        # Wrap all tasks with semaphore
        wrapped_tasks = [run_with_semaphore(task) for task in tasks]

        # Execute all wrapped tasks
        # Since wrapped_tasks are coroutines, we need to run them directly
        # Use nursery concept from anyio
        results = []

        async def collect_result(task: Coroutine) -> None:
            result = await task
            results.append(result)

        async with anyio.create_task_group() as tg:
            for task in wrapped_tasks:
                tg.start_soon(collect_result, task)

        return results

    def _extract_round_metrics(self, rounds: List[ConsensusRound[T]]) -> List[RoundMetric]:
        """Extract metrics from each round for state tracking."""
        metrics: List[RoundMetric] = []
        for round in rounds:
            round_metric = RoundMetric(
                round_number=round.round_number,
                convergence_score=self._calculate_convergence_score([round]),
                consensus_achieved=round.consensus_achieved,
                num_responses=len(round.responses),
                unique_votes=(
                    len(set(self._get_voting_group(r.content) for r in round.responses)) if round.responses else 0
                ),
            )
            metrics.append(round_metric)
        return metrics

    def _log_final_consensus_result(self, result: ConsensusResult[T], start_time: datetime) -> None:
        """Log comprehensive final consensus result - single output with all key information."""
        if self._settings.verbosity == VerbosityLevel.SILENT:
            return

        duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        # Determine result status and formatting
        if result.consensus_achieved:
            status_icon = "✅"
            status_text = "SUCCESS"
            result_display = status_text  # No emoji in table
        else:
            status_icon = "⚠️"
            status_text = "FALLBACK"
            result_display = status_text  # No emoji in table

        # Build comprehensive log message
        log_lines = []
        log_lines.append("")  # Empty line for spacing
        log_lines.append("=" * 80)
        log_lines.append(f"{status_icon}  CONSENSUS {status_text}")
        log_lines.append("=" * 80)

        # Core metrics as a table
        log_lines.append("")  # Add spacing
        log_lines.append("📊  SUMMARY:")

        table_data = [
            ["Result", result_display],
            ["Duration", f"{duration_ms:.1f}ms"],
            ["Rounds", str(result.total_rounds)],
            ["Models", str(len(result.participating_models))],
            ["Convergence", f"{result.convergence_score:.3f}"],
        ]

        # Voting details
        if result.rounds:
            final_round = result.rounds[-1]
            unique_votes = len(set(self._get_voting_group(r.content) for r in final_round.responses))

            # Build detailed vote groups with actual response content
            vote_groups_detailed = {}
            for response in final_round.responses:
                vote_key = self._get_voting_group(response.content)
                if vote_key not in vote_groups_detailed:
                    vote_groups_detailed[vote_key] = {
                        "models": [],
                        "response": response.content,
                    }
                vote_groups_detailed[vote_key]["models"].append(response.id)

            # Sort groups by size
            sorted_groups = sorted(
                vote_groups_detailed.items(),
                key=lambda x: len(x[1]["models"]),
                reverse=True,
            )
            top_group_size = len(sorted_groups[0][1]["models"]) if sorted_groups else 0
            agreement = top_group_size / len(final_round.responses) if final_round.responses else 0

            # Calculate field consistency and vote stability
            field_consistency = self._calculate_field_consistency(result.rounds)
            vote_stability = self._calculate_vote_stability(result.rounds)

            # Add voting metrics to the table
            table_data.extend(
                [
                    [
                        "Agreement",
                        f"{agreement:.1%} ({top_group_size}/{len(final_round.responses)})",
                    ],
                    ["Field Consistency", f"{field_consistency:.1%}"],
                    ["Vote Stability", f"{vote_stability:.1%}"],
                    ["Unique Votes", str(unique_votes)],
                    ["Threshold", f"{self._settings.threshold:.1%}"],
                ]
            )

        # Add total model calls if available
        if result.metrics:
            table_data.append(["Total Model Calls", str(result.metrics.total_model_calls)])

        # Format the table using regular string formatting
        # Find max width for proper alignment
        max_metric_width = max(len("Metric"), max(len(row[0]) for row in table_data))
        max_value_width = max(len("Value"), max(len(str(row[1])) for row in table_data))

        # Add header
        log_lines.append(f"   {'Metric':<{max_metric_width}}  {'Value':<{max_value_width}}")
        log_lines.append(f"   {'-' * max_metric_width}  {'-' * max_value_width}")

        # Add data rows
        for metric, value in table_data:
            log_lines.append(f"   {metric:<{max_metric_width}}  {value:<{max_value_width}}")

        # Detailed voting breakdown for all rounds (or just final if only one)
        if result.rounds:
            total_rounds = result.total_rounds

            # If multiple rounds, show evolution; otherwise just show the single round
            rounds_to_show = result.rounds if total_rounds > 1 else [result.rounds[-1]]

            for round_idx, round_data in enumerate(rounds_to_show):
                if not round_data.responses:
                    continue

                # Build vote groups for this round
                vote_groups_for_round = {}
                for response in round_data.responses:
                    vote_key = self._get_voting_group(response.content)
                    if vote_key not in vote_groups_for_round:
                        vote_groups_for_round[vote_key] = {
                            "models": [],
                            "response": response.content,
                        }
                    vote_groups_for_round[vote_key]["models"].append(response.id)

                # Sort groups by size
                sorted_round_groups = sorted(
                    vote_groups_for_round.items(),
                    key=lambda x: len(x[1]["models"]),
                    reverse=True,
                )

                log_lines.append("")  # Add spacing
                # Show which round this voting is from (1-indexed for display)
                display_round = round_idx + 1
                log_lines.append(f"🗳️  VOTING BREAKDOWN (Round {display_round}/{total_rounds}):")

                for i, (vote_key, vote_info) in enumerate(sorted_round_groups, 1):
                    models_str = ", ".join(vote_info["models"])
                    vote_count = len(vote_info["models"])
                    percentage = (vote_count / len(round_data.responses)) * 100

                    # Get a summary of what they voted for
                    response_content = vote_info["response"]
                    vote_summary = self._get_vote_summary(response_content)

                    winner_indicator = "👑" if i == 1 else "  "
                    log_lines.append(
                        f"   {winner_indicator} Vote #{i}: {vote_count} models ({percentage:.1f}%) - {models_str}"
                    )
                    log_lines.append(f"      └─ {vote_summary}")

        # Model Contributions Table
        if result.model_contributions:
            log_lines.append("")  # Add spacing
            log_lines.append("🤝  MODEL CONTRIBUTIONS (Score Range: 0.0-1.0):")
            log_lines.append("   Score Interpretation: ≥0.8=High, ≥0.6=Medium, ≥0.4=Low, <0.4=Very Low")

            # Prepare contribution data for table
            contrib_data = []
            for model_id, score in sorted(result.model_contributions.items(), key=lambda x: x[1], reverse=True):
                # Classify contribution level
                if score >= 0.8:
                    level = "High"
                elif score >= 0.6:
                    level = "Medium"
                elif score >= 0.4:
                    level = "Low"
                else:
                    level = "Very Low"
                contrib_data.append([model_id, f"{score:.3f}", level])

            if contrib_data:
                # Format contribution table using regular string formatting
                max_model_width = max(len("Model"), max(len(row[0]) for row in contrib_data))
                max_score_width = max(len("Score (0-1)"), max(len(row[1]) for row in contrib_data))
                max_level_width = max(len("Level"), max(len(row[2]) for row in contrib_data))

                # Add header
                log_lines.append(
                    f"   {'Model':<{max_model_width}}  {'Score (0-1)':<{max_score_width}}  {'Level':<{max_level_width}}"
                )
                log_lines.append(f"   {'-' * max_model_width}  {'-' * max_score_width}  {'-' * max_level_width}")

                # Add data rows
                for model, score, level in contrib_data:
                    log_lines.append(
                        f"   {model:<{max_model_width}}  {score:<{max_score_width}}  {level:<{max_level_width}}"
                    )

        # Response Evolution Summary (for multi-round consensus)
        if result.total_rounds > 1 and result.rounds:
            log_lines.append("")  # Add spacing
            log_lines.append("🔄  RESPONSE EVOLUTION:")

            evolution_data = []
            # Track which models participated in each round
            for round_idx, round_data in enumerate(result.rounds):
                round_num = round_idx + 1  # 1-indexed for display

                if round_idx == 0:
                    # Round 1: All models start fresh (no evolution yet)
                    for response in round_data.responses:
                        evolution_data.append([f"Round {round_num}", response.id, "Initial response"])
                else:
                    # Subsequent rounds: Check for evolution
                    if round_data.response_evolutions:
                        for evolution in round_data.response_evolutions:
                            changes = []
                            if evolution.vote_changed:
                                changes.append("Vote Changed")
                            if evolution.field_changes:
                                changes.append(f"{len(evolution.field_changes)} fields modified")

                            if changes:
                                evolution_data.append(
                                    [
                                        f"Round {round_num}",
                                        evolution.id,
                                        ", ".join(changes),
                                    ]
                                )
                            else:
                                evolution_data.append([f"Round {round_num}", evolution.id, "No changes"])
                    else:
                        # No evolutions tracked for this round
                        for response in round_data.responses:
                            evolution_data.append(
                                [
                                    f"Round {round_num}",
                                    response.id,
                                    "No changes tracked",
                                ]
                            )

            if evolution_data:
                # Format evolution table using regular string formatting
                max_round_width = max(len("Round"), max(len(row[0]) for row in evolution_data))
                max_model_width = max(len("Model"), max(len(row[1]) for row in evolution_data))
                max_changes_width = max(len("Changes"), max(len(row[2]) for row in evolution_data))

                # Add header
                log_lines.append(
                    f"   {'Round':<{max_round_width}}  {'Model':<{max_model_width}}  {'Changes':<{max_changes_width}}"
                )
                log_lines.append(f"   {'-' * max_round_width}  {'-' * max_model_width}  {'-' * max_changes_width}")

                # Add data rows
                for round_str, model, changes in evolution_data:
                    log_lines.append(
                        f"   {round_str:<{max_round_width}}  {model:<{max_model_width}}  {changes:<{max_changes_width}}"
                    )
            else:
                log_lines.append("   No evolution data available")

        # Information Flow (if tracked)
        if result.rounds:
            has_flow = any(r.information_flow for r in result.rounds)
            if has_flow or result.total_rounds > 1:
                log_lines.append("")  # Add spacing
                log_lines.append("📡  INFORMATION FLOW:")

                flow_data = []
                for round_idx, round_data in enumerate(result.rounds):
                    round_num = round_idx + 1  # 1-indexed for display

                    if round_idx == 0:
                        # Round 1: No prior influence (models start independently)
                        for response in round_data.responses:
                            flow_data.append(
                                [
                                    f"Round {round_num}",
                                    response.id,
                                    "Independent (no prior rounds)",
                                ]
                            )
                    elif round_data.information_flow:
                        # Subsequent rounds with tracked influence
                        for (
                            influenced_model,
                            influencers,
                        ) in round_data.information_flow.items():
                            if influencers:
                                flow_data.append(
                                    [
                                        f"Round {round_num}",
                                        influenced_model,
                                        " → ".join(influencers),
                                    ]
                                )
                            else:
                                flow_data.append(
                                    [
                                        f"Round {round_num}",
                                        influenced_model,
                                        "No direct influence",
                                    ]
                                )
                    else:
                        # No flow data for this round but show models participated
                        for response in round_data.responses:
                            flow_data.append(
                                [
                                    f"Round {round_num}",
                                    response.id,
                                    "Influence not tracked",
                                ]
                            )

                if flow_data:
                    # Format flow table using regular string formatting
                    max_round_width = max(len("Round"), max(len(row[0]) for row in flow_data))
                    max_model_width = max(len("Model"), max(len(row[1]) for row in flow_data))
                    max_influenced_width = max(len("Influenced By"), max(len(row[2]) for row in flow_data))

                    # Add header
                    log_lines.append(
                        f"   {'Round':<{max_round_width}}  {'Model':<{max_model_width}}  {'Influenced By':<{max_influenced_width}}"
                    )
                    log_lines.append(
                        f"   {'-' * max_round_width}  {'-' * max_model_width}  {'-' * max_influenced_width}"
                    )

                    # Add data rows
                    for round_str, model, influenced in flow_data:
                        log_lines.append(
                            f"   {round_str:<{max_round_width}}  {model:<{max_model_width}}  {influenced:<{max_influenced_width}}"
                        )

        # Final response info
        if result.final_response:
            log_lines.append("")  # Add spacing
            log_lines.append("🎯  FINAL RESPONSE:")
            # Show key fields from the response
            response_dict = result.final_response.model_dump()
            for key, value in response_dict.items():
                if key == "reasoning":
                    continue  # Show reasoning separately below
                # Wrap long values for readability
                value_str = str(value)
                if len(value_str) > 60:
                    wrapped_value = self._wrap_text_for_logging(value_str, max_length=60, indent=4)
                    log_lines.append(f"   • {key}:")
                    for line in wrapped_value.split("\n"):
                        log_lines.append(f"     {line}")
                else:
                    log_lines.append(f"   • {key}: {value_str}")

        # Disagreement analysis
        if result.rounds and result.rounds[-1].disagreement_analysis:
            disagreement = result.rounds[-1].disagreement_analysis
            if disagreement.disagreement_fields:
                log_lines.append("")  # Add spacing
                log_lines.append(f"⚡  DISAGREEMENTS: {list(disagreement.disagreement_fields.keys())}")
            if disagreement.consensus_fields:
                log_lines.append(f"🤝  CONSENSUS FIELDS: {disagreement.consensus_fields}")

        # Reasoning - show beautifully wrapped
        if result.reasoning:
            log_lines.append("")  # Add spacing
            log_lines.append("💭  CONSENSUS REASONING:")
            wrapped_reasoning = self._wrap_text_for_logging(result.reasoning, max_length=76, indent=3)
            for line in wrapped_reasoning.split("\n"):
                log_lines.append(f"   {line}")

        log_lines.append("=" * 80)

        # Log everything as a single block
        logger.info("\n" + "\n".join(log_lines))

    def _get_vote_summary(self, response_content: T) -> str:
        """Generate a detailed summary of what a vote group voted for, including reasoning."""
        try:
            # Convert to dict to inspect fields
            content_dict = response_content.model_dump()

            # Separate voting fields and ignored/reasoning fields
            voting_fields = []
            ignored_fields = []
            reasoning_text = ""

            for field_name, field_value in content_dict.items():
                # Handle reasoning specially
                if field_name == "reasoning":
                    reasoning_text = str(field_value)
                    continue

                # Get field metadata to check if it's ignored
                field_info = response_content.__class__.model_fields.get(field_name)
                is_ignored = False
                if field_info and field_info.json_schema_extra:
                    extra = field_info.json_schema_extra
                    if isinstance(extra, dict) and "voting_comparison" in extra:
                        voting_comparison = extra["voting_comparison"]
                        if isinstance(voting_comparison, dict):
                            strategy = voting_comparison.get("strategy")
                            if strategy == "IGNORE" or (
                                hasattr(ComparisonStrategy, "IGNORE") and strategy == ComparisonStrategy.IGNORE.value
                            ):
                                is_ignored = True

                # Format the field value nicely
                if isinstance(field_value, (str, int, float, bool)):
                    field_str = f"{field_name}={field_value}"
                else:
                    field_str = f"{field_name}={type(field_value).__name__}"

                if is_ignored:
                    ignored_fields.append(f"({field_str})")  # Parentheses indicate ignored
                else:
                    voting_fields.append(field_str)

            # Build the summary
            summary_parts = []

            # Add voting fields first
            if voting_fields:
                summary_parts.append(", ".join(voting_fields))

            # Add ignored fields with indicator
            if ignored_fields:
                summary_parts.append(f"🔒 {', '.join(ignored_fields)}")

            # Add reasoning with wrapping
            if reasoning_text:
                # Wrap reasoning to multiple lines with proper indentation
                wrapped_reasoning = self._wrap_text_for_logging(reasoning_text, max_length=70, indent=10)
                summary_parts.append(f"💭 {wrapped_reasoning}")

            return "\n      ".join(summary_parts) if summary_parts else "No fields found"

        except Exception:
            # Safe fallback
            return str(response_content)

    def _calculate_field_similarity(self, values: List[str]) -> float:
        """Calculate semantic similarity score for a set of field values.

        Returns a score between 0 (completely different) and 1 (identical).
        Uses semantic embeddings to determine actual similarity between values.
        """
        if not values or len(values) == 1:
            return 1.0

        unique_values = list(set(values))
        if len(unique_values) == 1:
            return 1.0

        # Use semantic similarity for text values
        try:
            # Get embeddings for all unique values
            embeddings = []
            for value in unique_values:
                # Truncate very long values for embedding
                truncated = value[:1000] if len(value) > 1000 else value
                embedding = EncoderCore.encode_single(truncated)
                embeddings.append(embedding)

            # Calculate pairwise similarities
            import numpy as np

            similarities = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    # Cosine similarity
                    similarity = np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                    )
                    similarities.append(similarity)

            if similarities:
                # Average similarity across all pairs
                avg_similarity = float(np.mean(similarities))

                # Weight by how many models agree
                # If most models agree on one value, boost the score
                value_counts = Counter(values)
                max_count = max(value_counts.values())
                agreement_ratio = max_count / len(values)

                # Combine semantic similarity with agreement ratio
                # If values are semantically similar OR most models agree, score is high
                final_score = max(avg_similarity, agreement_ratio * 0.8)

                return final_score

        except Exception as e:
            logger.debug(f"Semantic similarity calculation failed, using basic method: {e}")

        # Fallback to basic calculation
        similarity = 1.0 - ((len(unique_values) - 1) / len(values))
        return similarity
