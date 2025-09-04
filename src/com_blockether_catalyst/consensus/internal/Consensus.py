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
from pydantic import BaseModel, Field

from ...encoder import EncoderCore
from ...utils.TypedCalls import ArityOneTypedCall
from .ConsensusTypes import (
    ConsensusCallRecord,
    ConsensusMetrics,
    ConsensusResult,
    ConsensusRound,
    ConsensusSettings,
    ConsensusState,
    DisagreementAnalysis,
    FieldChangeValue,
    GossipHistory,
    ModelConfiguration,
    ModelMetrics,
    ModelResponse,
    ResponseEvolution,
    ResponseMetadata,
    TypedCallBaseForConsensus,
    VerbosityLevel,
)

# Type variable for structured outputs
T = TypeVar("T", bound=TypedCallBaseForConsensus)

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

        # State management
        self._metrics_collector: DefaultDict[str, List[Union[str, int, float]]] = defaultdict(list)
        self._model_metrics: Dict[str, ModelMetrics] = {}
        self._model_response_history: DefaultDict[str, List[ModelResponse[T]]] = defaultdict(list)
        self._response_evolution_tracking: DefaultDict[str, List[ResponseEvolution[T]]] = defaultdict(list)

        # Initialize consensus state tracking
        self._state = ConsensusState[T](
            max_history_size=self._settings.max_history_size,
            enabled=self._settings.state_tracking,
        )

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

        # Initial round - get independent responses
        logger.info("🚀 === INITIAL CONSENSUS ROUND 0 ===")
        logger.debug(f"Starting consensus for query: {x[:100]}...")
        initial_round = await self._execute_initial_round(x)
        rounds.append(initial_round)

        # Log initial round metrics
        self._log_round_metrics(initial_round, 0)

        # Log initial responses
        if initial_round.responses:
            logger.info(f"📝 Initial responses ({len(initial_round.responses)} models):")
            for i, resp in enumerate(initial_round.responses, 1):
                logger.info(f"  {i}. {resp.id}: {str(resp.content)}")

        # Check for early consensus
        if await self._check_consensus(initial_round, convergence_threshold):
            assert (
                initial_round.consensus_response is not None
            ), "Consensus response must be set when consensus is achieved"
            return self._create_result(
                rounds,
                initial_round.consensus_response,
                start_time,
            )

        # Iterative refinement rounds
        for round_num in range(1, max_rounds):
            logger.info(f"🔄 === CONSENSUS ROUND {round_num} ===")

            # Log current responses from previous round
            if rounds and rounds[-1].responses:
                logger.info(f"📝 Current responses ({len(rounds[-1].responses)} models):")
                for i, resp in enumerate(rounds[-1].responses, 1):
                    logger.info(f"  {i}. {resp.id}: {resp.content}")

            logger.debug(f"Executing consensus round {round_num}")

            # Execute gossip round
            consensus_round = await self._execute_gossip_round(x, rounds, round_num)
            rounds.append(consensus_round)

            # Log inter-round metrics if verbose
            self._log_round_metrics(consensus_round, round_num)

            # Check for consensus
            if await self._check_consensus(consensus_round, convergence_threshold):
                assert (
                    consensus_round.consensus_response is not None
                ), "Consensus response must be set when consensus is achieved"
                return self._create_result(
                    rounds,
                    consensus_round.consensus_response,
                    start_time,
                )

        # Fallback to majority vote
        logger.warning("Consensus not reached, falling back to majority vote")
        # Get response type from the first response in the last round
        response_type = type(rounds[-1].responses[0].content) if rounds[-1].responses else None
        if not response_type:
            raise ValueError("Cannot determine response type - no responses available")
        final_response = await self._majority_vote(rounds[-1], response_type)

        result = self._create_result(rounds, final_response, start_time)

        # Track this call in state if enabled
        if self._settings.state_tracking:
            duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            convergence_path = [self._calculate_convergence_score(rounds[: i + 1]) for i in range(len(rounds))]

            call_record = ConsensusCallRecord[T](
                timestamp=start_time,
                input_prompt=x,
                result=result,
                round_metrics=self._extract_round_metrics(rounds),
                convergence_path=convergence_path,
                duration_ms=duration_ms,
            )
            self._state.add_call(call_record)

        return result

    async def _execute_initial_round(self, query: str) -> ConsensusRound[T]:
        """Execute the initial round where models respond independently."""
        responses = []

        # Create tasks for parallel execution with concurrency control
        tasks = []
        model_configs = self._models

        for model_config in model_configs:
            prompt = f"{model_config.perspective}\n\n{query}"
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
                self._response_evolution_tracking[model_config.id].append(evolution)
            except Exception as e:
                logger.error(f"Model {model_config.id} processing failed: {e}")

        # Collect evolutions for this round
        round_evolutions = []
        for id, evolutions in self._response_evolution_tracking.items():
            for evolution in evolutions:
                if evolution.round_to == round_num:
                    round_evolutions.append(evolution)

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
                prompt += f"✓ Consensus reached on: {', '.join(analysis.consensus_fields[:5])}\n"

            if analysis.disagreement_fields:
                disagreement_field_names = list(analysis.disagreement_fields.keys())[:3]
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
        logger.debug(f"Checking consensus for round {round_data.round_number}, responses: {len(round_data.responses)}")
        if len(round_data.responses) < 2:
            logger.debug("Only one response - consensus achieved")
            if round_data.responses:
                round_data.consensus_achieved = True
                round_data.consensus_response = round_data.responses[0].content
            return True

        responses = round_data.responses
        logger.info(f"\n🗳️ === VOTING ROUND {round_data.round_number} ===")
        logger.info(f"   Counting votes from {len(responses)} models")

        # Count votes by hashing responses
        vote_counts: Dict[str, List[ModelResponse[T]]] = {}
        for response in responses:
            # Create a hash of the response for voting
            response_hash = self._hash_response(response.content)
            if response_hash not in vote_counts:
                vote_counts[response_hash] = []
            vote_counts[response_hash].append(response)

        # Log vote distribution
        for hash_key, voters in vote_counts.items():
            model_ids = [v.id for v in voters]
            logger.info(f"   📊 Vote group ({len(voters)} votes): {', '.join(model_ids)}")

        # Check if we have a clear majority
        total_votes = len(responses)
        majority_threshold = total_votes / 2.0

        # Sort vote groups by count
        sorted_votes = sorted(vote_counts.items(), key=lambda x: len(x[1]), reverse=True)
        top_vote_count = len(sorted_votes[0][1])

        # Check for clear majority
        if top_vote_count > majority_threshold:
            # Clear majority winner
            winner = sorted_votes[0][1][0]  # Take first response from winning group
            logger.info(f"   ✅ MAJORITY CONSENSUS! {top_vote_count}/{total_votes} votes")
            logger.info(f"   🎯 Winner: {winner.id}")
            round_data.consensus_achieved = True
            round_data.consensus_response = winner.content
            return True

        # Check if we need the judge for a tie
        if len(sorted_votes) > 1 and len(sorted_votes[0][1]) == len(sorted_votes[1][1]):
            logger.info(f"   ⚖️ TIE detected: {len(sorted_votes[0][1])} votes each")
            # We have a tie - consensus not achieved yet, will continue rounds
            return False

        # Plurality winner (not majority but leading)
        if top_vote_count >= threshold * total_votes:
            # Accept plurality if it meets threshold
            winner = sorted_votes[0][1][0]
            logger.info(f"   ✓ PLURALITY CONSENSUS! {top_vote_count}/{total_votes} votes (>={threshold:.0%} threshold)")
            logger.info(f"   🎯 Winner: {winner.id}")
            round_data.consensus_achieved = True
            round_data.consensus_response = winner.content
            return True

        logger.info(f"   ❌ No consensus - top vote: {top_vote_count}/{total_votes} votes")
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

        # Count votes by hashing responses
        vote_counts: Dict[str, List[ModelResponse[T]]] = {}
        for response in responses:
            response_hash = self._hash_response(response.content)
            if response_hash not in vote_counts:
                vote_counts[response_hash] = []
            vote_counts[response_hash].append(response)

        # Sort by vote count
        sorted_votes = sorted(vote_counts.items(), key=lambda x: len(x[1]), reverse=True)

        # Check for tie at the top
        if len(sorted_votes) > 1 and len(sorted_votes[0][1]) == len(sorted_votes[1][1]):
            # We have a tie - use judge to resolve it
            logger.info("⚖️ Using judge to resolve tie...")
            tied_responses = [sorted_votes[0][1][0], sorted_votes[1][1][0]]
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
        # Prepare judge prompt with voting history and tied responses
        judge_prompt = f"""As a neutral judge, you must break the tie between these responses and provide the best synthesis.

## Voting History (Gossips):
Round {round_data.round_number} had {len(round_data.responses)} total votes.

## Tied Responses:

Response 1 (from {tied_responses[0].id}):
{tied_responses[0].content.model_dump_json(indent=2)}

Response 2 (from {tied_responses[1].id}):
{tied_responses[1].content.model_dump_json(indent=2)}

## Your Task:
Analyze both responses and provide YOUR OWN response that either:
1. Selects the better of the two responses
2. Synthesizes the best elements from both
3. Provides an improved answer based on their insights

Base your decision on:
- Quality of reasoning
- Internal consistency
- Supporting evidence
- Completeness of answer

Provide a response in the same JSON format as the tied responses above.
"""

        # Call the judge - it returns type T directly
        judge_response = await self._judge.call(judge_prompt)

        # Create a new ModelResponse with the judge's decision
        logger.info("   ⚖️ Judge provided synthesis/selection")
        return ModelResponse[T](
            id="judge",
            round_number=round_data.round_number,
            content=judge_response,  # judge_response is already of type T
            metadata=ResponseMetadata(judge_decision=True, resolved_tie=True),
            gossip_history=[
                GossipHistory(
                    round_number=round_data.round_number,
                    refined_from_peers=True,
                    peer_models_seen=[r.id for r in tied_responses],
                )
            ],
        )

    def _create_result(
        self,
        rounds: List[ConsensusRound[T]],
        final_response: T,
        start_time: datetime,
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
        )

        # Generate reasoning based on the consensus process
        reasoning = self._generate_consensus_reasoning(rounds, final_round.consensus_achieved, dissenting_models)

        return ConsensusResult(
            reasoning=reasoning,
            consensus_achieved=final_round.consensus_achieved,
            final_response=final_response,
            rounds=rounds,
            total_rounds=len(rounds),
            convergence_score=self._calculate_convergence_score(rounds),
            participating_models=[m.id for m in self._models],
            dissenting_models=dissenting_models,
            metrics=metrics,
        )

    def _generate_consensus_reasoning(
        self,
        rounds: List[ConsensusRound[T]],
        consensus_achieved: bool,
        dissenting_models: List[str],
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
            return f"Consensus was not achieved after {len(rounds)} round(s) of deliberation{dissent_info}. The system fell back to majority voting to determine the final response. Despite the lack of full agreement, this represents the best collective judgment available."

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
        vote_hashes = set()
        for response in responses:
            response_hash = self._hash_response(response.content)
            vote_hashes.add(response_hash)

        # Agreement ratio: 1.0 if all same, approaching 0 if all different
        agreement_ratio = 1.0 - ((len(vote_hashes) - 1) / (len(responses) - 1))
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
                self._model_response_history[response.id].append(response)

        # Calculate contribution metrics for each model
        for model_id, responses in model_responses.items():
            metrics = self._model_metrics.get(model_id, ModelMetrics(id=model_id))

            # Update participation count
            metrics.total_rounds += len(responses)

            # Calculate consistency across rounds
            metrics.consistency_score = self._calculate_consistency_score(responses)

            # Calculate contribution score
            contribution_score = self._calculate_contribution_score(responses, rounds)
            metrics.contribution_score = contribution_score

            model_contributions[model_id] = contribution_score
            self._model_metrics[model_id] = metrics

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
        final_hash = self._hash_response(final_consensus)
        voted_correctly = 0

        for response in model_responses:
            response_hash = self._hash_response(response.content)
            if response_hash == final_hash:
                voted_correctly += 1

        # Score based on how often model voted for consensus
        base_score = voted_correctly / len(model_responses)

        # Bonus if model converged to consensus over time
        if len(model_responses) >= 2:
            early_match = self._hash_response(model_responses[0].content) == final_hash
            late_match = self._hash_response(model_responses[-1].content) == final_hash
            if late_match and not early_match:
                # Model learned and converged
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
            response_hash = self._hash_response(r.content)
            vote_counts[response_hash] = vote_counts.get(response_hash, 0) + 1

        # Get this response's vote group size
        response_hash = self._hash_response(response.content)
        response_vote_count = vote_counts.get(response_hash, 0)

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
        vote_hashes = []
        for response in responses:
            vote_hashes.append(self._hash_response(response.content))

        # Count how many times the vote stayed the same between rounds
        consistent_votes = 0
        for i in range(len(vote_hashes) - 1):
            if vote_hashes[i] == vote_hashes[i + 1]:
                consistent_votes += 1

        return consistent_votes / (len(vote_hashes) - 1) if len(vote_hashes) > 1 else 1.0

    # Enhanced metrics methods
    def _calculate_consensus_confidence(
        self,
        consensus_achieved: bool,
        total_rounds: int,
        dissenting_count: int,
        total_models: int,
    ) -> float:
        """Calculate voting strength as confidence metric."""
        if not consensus_achieved:
            return 0.0

        # For voting: confidence is based on vote proportion
        # If all models agree: 1.0
        # If bare majority: lower confidence
        voting_proportion = (total_models - dissenting_count) / max(total_models, 1)

        # Adjust for speed - faster consensus = stronger agreement
        speed_bonus = (
            max(
                0,
                (self._settings.max_rounds - total_rounds) / self._settings.max_rounds,
            )
            * 0.2
        )

        return min(1.0, voting_proportion + speed_bonus)

    # Virtual Voting Methods
    @staticmethod
    def _semantic_hash(text: str, threshold: float = 0.8) -> str:
        """
        Create a hash that groups semantically similar texts.

        This is used for voting - texts with high semantic similarity
        get the same hash and thus vote together.

        Args:
            text: Text to hash
            threshold: Similarity threshold for grouping

        Returns:
            Hash string that's identical for semantically similar texts
        """
        try:
            # Normalize text first - lowercase and strip for consistent hashing
            normalized_text = text.lower().strip()

            # Get embedding for the normalized text
            embedding = EncoderCore.encode_single(normalized_text)

            # Quantize embedding to create discrete buckets
            # This ensures similar texts map to the same bucket
            import numpy as np

            # Scale and quantize based on threshold
            # Higher threshold = finer quantization = fewer matches
            quantization_level = int(10 / threshold)
            quantized = np.round(embedding * quantization_level).astype(int)

            # Create hash from quantized vector
            return hashlib.md5(quantized.tobytes()).hexdigest()[:8]
        except Exception as e:
            logger.debug(f"Semantic hashing failed, falling back to simple hash: {e}")
            # Fallback: case-insensitive hash with same normalization
            normalized_text = text.lower().strip()
            return hashlib.md5(normalized_text.encode()).hexdigest()[:8]

    def _hash_response(self, response: T) -> str:
        """Create a deterministic hash of a response for voting using field comparison strategies.

        Now delegates to the response type's own get_voting_key method.
        """
        return response.get_voting_key()

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

        # Get hashes for vote comparison
        prev_hash = self._hash_response(prev_response.content)
        new_hash = self._hash_response(new_response.content)

        # Check if vote changed
        vote_changed = prev_hash != new_hash

        # Extract reasoning evolution
        reasoning_evolution = ""
        if "reasoning" in prev_dict and "reasoning" in new_dict:
            if prev_dict["reasoning"] != new_dict["reasoning"]:
                reasoning_evolution = (
                    f"Changed from: {prev_dict['reasoning'][:100]}... to: {new_dict['reasoning'][:100]}..."
                )

        # Determine which models influenced this evolution
        influenced_by = []
        # Check if model changed vote to match any peer (using hashes calculated above)

        if prev_hash != new_hash:  # Vote changed
            for peer in peer_responses:
                peer_hash = self._hash_response(peer.content)
                if new_hash == peer_hash:
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
        """Analyze disagreements between responses."""
        if not responses:
            return DisagreementAnalysis()

        analysis = DisagreementAnalysis()

        # Collect all field values as strings
        field_values: DefaultDict[str, List[str]] = defaultdict(list)
        for response in responses:
            response_dict = response.content.model_dump()
            for field, value in response_dict.items():
                field_values[field].append(str(value))

        # Analyze each field
        for field, values in field_values.items():
            unique_values = list(set(values))

            if len(unique_values) == 1:
                # Consensus on this field
                analysis.consensus_fields.append(field)
            else:
                # Disagreement on this field
                analysis.disagreement_fields[field] = values

        return analysis

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

    def _extract_round_metrics(self, rounds: List[ConsensusRound[T]]) -> List[Dict[str, Any]]:
        """Extract metrics from each round for state tracking."""
        metrics = []
        for round in rounds:
            round_metric = {
                "round_number": round.round_number,
                "convergence_score": self._calculate_convergence_score([round]),
                "consensus_achieved": round.consensus_achieved,
                "num_responses": len(round.responses),
                "unique_votes": (
                    len(set(self._get_voting_key_from_content(r.content) for r in round.responses))
                    if round.responses
                    else 0
                ),
            }
            metrics.append(round_metric)
        return metrics

    def reset_state(self) -> None:
        """Reset the consensus state history."""
        self._state.reset()
        self._log_based_on_verbosity("State history has been reset", VerbosityLevel.VERBOSE)

    def get_state(self) -> ConsensusState[T]:
        """Get the current consensus state object."""
        return self._state

    def get_state_stats(self) -> Dict[str, Any]:
        """Get statistics about the consensus state."""
        return self._state.get_stats()

    def _log_based_on_verbosity(self, message: str, min_level: VerbosityLevel) -> None:
        """Log a message if verbosity level permits."""
        if self._settings.verbosity.value >= min_level.value:
            logger.info(message)

    def _get_voting_key_from_content(self, content: T) -> str:
        """Generate a voting key from content for comparison."""
        # T is bound to TypedCallBaseForConsensus which has get_voting_key method
        return content.get_voting_key()

    def _log_round_metrics(self, round: ConsensusRound[T], round_num: int) -> None:
        """Log metrics for a consensus round based on verbosity level."""
        if self._settings.verbosity == VerbosityLevel.SILENT:
            return

        # Basic logging for NORMAL level
        if self._settings.verbosity.value >= VerbosityLevel.NORMAL.value:
            unique_votes = len(set(self._get_voting_key_from_content(r.content) for r in round.responses))
            convergence = self._calculate_convergence_score([round])
            logger.info(
                f"Round {round_num}: {len(round.responses)} responses, "
                f"{unique_votes} unique votes, convergence: {convergence:.2f}"
            )

        # Detailed logging for VERBOSE level
        if self._settings.verbosity == VerbosityLevel.VERBOSE:
            # Log voting distribution
            vote_counts: dict[str, int] = {}
            for response in round.responses:
                key = self._get_voting_key_from_content(response.content)
                vote_counts[key] = vote_counts.get(key, 0) + 1

            logger.info(f"  Voting distribution: {dict(sorted(vote_counts.items(), key=lambda x: x[1], reverse=True))}")

            if round.consensus_achieved:
                logger.info(f"  ✓ Consensus achieved in round {round_num}")

            # Log disagreement analysis if present
            if round.disagreement_analysis:
                num_disagreements = len(round.disagreement_analysis.disagreement_fields)
                logger.info(f"  Disagreements: {num_disagreements} fields")
                if round.disagreement_analysis.disagreement_fields:
                    for (
                        field,
                        values,
                    ) in round.disagreement_analysis.disagreement_fields.items():
                        logger.info(f"    - {field}: {len(values)} unique values")
