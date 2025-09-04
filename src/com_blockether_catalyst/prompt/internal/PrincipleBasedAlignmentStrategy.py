"""
Principle-based alignment strategy for prompt optimization.

This module implements the superior principle-based approach for prompt alignment,
which extracts reusable guidelines from feedback and systematically applies them.

Based on the model alignment techniques demonstrated in Google's Gemma notebook:
https://colab.research.google.com/github/pair-code/model-alignment/blob/main/notebooks/Gemma_for_Model_Alignment.ipynb

The key insight from the notebook is that extracting principles (guidelines) from
feedback and applying them systematically leads to better long-term prompt quality
than directly modifying prompts based on immediate feedback.
"""

import re
from typing import List

from .PromptAlignmentTypes import AlignmentFeedback, AlignmentPrinciple


class PrincipleBasedAlignmentStrategy:
    """
    Strategy for aligning prompts using extracted principles.

    This is the recommended approach for prompt alignment as it:
    - Extracts generalizable principles from feedback
    - Applies improvements systematically
    - Builds reusable knowledge over time
    - Provides more stable and consistent results
    """

    # Static configuration
    MAX_PRINCIPLE_LENGTH = 200
    MIN_PRINCIPLE_LENGTH = 10
    MAX_PRINCIPLES_PER_FEEDBACK = 10

    def extract_principles(self, feedback: AlignmentFeedback) -> List[AlignmentPrinciple]:
        """
        Extract actionable principles from alignment feedback.

        This method identifies reusable guidelines that can be applied
        systematically to improve prompts, rather than making one-off changes.

        Args:
            feedback: The alignment feedback to extract principles from

        Returns:
            List of extracted principles ordered by importance
        """
        principles = []

        # Extract from explicit principles (highest quality)
        for principle in feedback.principles_to_apply.root:
            if self.MIN_PRINCIPLE_LENGTH <= len(principle.principle) <= self.MAX_PRINCIPLE_LENGTH:
                principles.append(principle)

        # Extract from improvement suggestions (actionable guidelines)
        for suggestion in feedback.improvement_suggestions.root[: self.MAX_PRINCIPLES_PER_FEEDBACK]:
            if self._is_actionable_principle(suggestion):
                principles.append(
                    AlignmentPrinciple(
                        principle=suggestion,
                        importance=0.8,
                    )
                )

        # Convert specific issues to positive principles
        for issue in feedback.specific_issues.root[: self.MAX_PRINCIPLES_PER_FEEDBACK]:
            positive_principle = self._convert_issue_to_principle(issue)
            if positive_principle:
                principles.append(
                    AlignmentPrinciple(
                        principle=positive_principle,
                        importance=0.9,
                    )
                )

        return principles[: self.MAX_PRINCIPLES_PER_FEEDBACK]

    def apply_principles(
        self,
        prompt: str,
        principles: List[AlignmentPrinciple],
        preserve_context: bool = True,
    ) -> str:
        """
        Apply principles to systematically transform a prompt.

        Principles are applied in order of importance to ensure the most
        critical improvements are made first.

        Args:
            prompt: The original prompt
            principles: Principles to apply, sorted by importance
            preserve_context: Whether to preserve original context

        Returns:
            Transformed prompt with principles applied
        """
        aligned_prompt = prompt

        # Sort by importance for systematic application
        sorted_principles = sorted(principles, key=lambda p: p.importance, reverse=True)

        for principle in sorted_principles:
            # Apply all principles using a unified approach
            aligned_prompt = self._apply_principle(aligned_prompt, principle, preserve_context)

        return aligned_prompt

    def _is_actionable_principle(self, text: str) -> bool:
        """Check if text represents an actionable, reusable principle."""
        actionable_keywords = [
            "should",
            "must",
            "ensure",
            "add",
            "remove",
            "include",
            "specify",
            "clarify",
            "provide",
            "always",
            "never",
            "require",
        ]
        return any(keyword in text.lower() for keyword in actionable_keywords)

    def _convert_issue_to_principle(self, issue: str) -> str:
        """Convert an identified issue to a positive, reusable principle."""
        issue_lower = issue.lower()

        if "lacks" in issue_lower:
            return issue.replace("lacks", "should include")
        elif "missing" in issue_lower:
            # Preserve original case when replacing
            if "Missing" in issue:
                return issue.replace("Missing", "should include")
            return issue.replace("missing", "should include")
        elif "unclear" in issue_lower:
            return "Ensure clarity and specificity in requirements"
        elif "ambiguous" in issue_lower:
            return "Provide specific, unambiguous instructions"
        elif "too vague" in issue_lower:
            return "Make requirements specific and concrete"
        elif "too long" in issue_lower:
            return "Keep prompts concise while maintaining clarity"
        elif "too complex" in issue_lower:
            return "Break down complex requirements into simpler parts"
        elif "no context" in issue_lower:
            return "Provide sufficient context for understanding"
        elif "no examples" in issue_lower:
            return "Include relevant examples to clarify expectations"

        # Generic conversion for other issues
        return f"Ensure: {issue.replace('not', '').replace('no', '').strip()}"

    def _apply_principle(self, prompt: str, principle: AlignmentPrinciple, preserve_context: bool) -> str:
        """Apply a principle to systematically improve the prompt."""
        principle_text = principle.principle.lower()

        # Clarity improvements
        if "clarity" in principle_text or "clear" in principle_text:
            if preserve_context:
                return f"{prompt}\n\nSpecific requirements: [Add specific details here]"
            else:
                return f"Clear instruction: {self._clarify_prompt(prompt)}"

        # Specificity improvements
        elif "specific" in principle_text or "concrete" in principle_text:
            if preserve_context:
                return f"{prompt}\n\nSpecifically: [Add concrete details]"
            else:
                return self._make_more_specific(prompt)

        # Conciseness improvements
        elif "concise" in principle_text or "shorten" in principle_text:
            return self._simplify_prompt(prompt)

        # Complexity reduction
        elif "simple" in principle_text or "break down" in principle_text:
            return self._break_down_complexity(prompt)

        # Context addition
        elif "context" in principle_text:
            return self._add_context(prompt, preserve_context)

        # Addition/inclusion improvements
        elif "add" in principle_text or "include" in principle_text:
            addition = self._extract_addition_content(principle.principle)
            if preserve_context:
                return f"{prompt}\n\n{addition}"
            else:
                return f"{addition}\n\n{prompt}"

        # Structure improvements
        elif "structure" in principle_text:
            return self._restructure_prompt(prompt)

        # Example additions
        elif "example" in principle_text:
            return f"{prompt}\n\nExample: [Add relevant example]"

        # Format improvements
        elif "format" in principle_text:
            return self._improve_formatting(prompt)

        # General principle application
        else:
            if preserve_context:
                return f"Principle: {principle.principle}\n\n{prompt}"
            else:
                return f"{principle.principle}\n\n{self._extract_core_request(prompt)}"

    def _clarify_prompt(self, prompt: str) -> str:
        """Clarify a prompt by making language more precise."""
        # Remove hedge words and make language direct
        clarified = prompt
        hedge_words = [
            "maybe",
            "perhaps",
            "might",
            "could",
            "possibly",
            "somewhat",
            "likely",
            "probably",
            "seems",
            "appears",
        ]
        for word in hedge_words:
            clarified = re.sub(rf"\b{word}\b", "", clarified, flags=re.IGNORECASE)

        # Clean up extra spaces
        clarified = " ".join(clarified.split())
        return clarified

    def _make_more_specific(self, prompt: str) -> str:
        """Make a prompt more specific with concrete markers."""
        if "?" in prompt:
            return prompt.replace("?", " with specific details and examples?")
        else:
            return f"{prompt} (provide specific, detailed response)"

    def _simplify_prompt(self, prompt: str) -> str:
        """Simplify a prompt by removing redundancy and wordiness."""
        # Remove duplicate consecutive words
        words = prompt.split()
        seen = set()
        simplified = []
        for word in words:
            word_lower = word.lower()
            # Keep articles and conjunctions even if repeated
            if word_lower not in seen or word_lower in {
                "the",
                "a",
                "an",
                "and",
                "or",
                "but",
                "to",
                "of",
                "in",
            }:
                simplified.append(word)
                if word_lower not in {
                    "the",
                    "a",
                    "an",
                    "and",
                    "or",
                    "but",
                    "to",
                    "of",
                    "in",
                }:
                    seen.add(word_lower)

        return " ".join(simplified)

    def _break_down_complexity(self, prompt: str) -> str:
        """Break down complex prompts into simpler parts."""
        # Split on conjunctions and list items
        parts = re.split(r"[;,]|\band\b", prompt)
        if len(parts) > 1:
            numbered_parts = [f"{i + 1}. {part.strip()}" for i, part in enumerate(parts) if part.strip()]
            return "Please address the following:\n" + "\n".join(numbered_parts)
        return prompt

    def _add_context(self, prompt: str, preserve_context: bool) -> str:
        """Add context to make the prompt more complete."""
        if preserve_context:
            return f"Context: [Provide relevant background]\n\n{prompt}"
        else:
            return f"Given the context of [specify domain/situation]:\n{prompt}"

    def _extract_addition_content(self, principle: str) -> str:
        """Extract what should be added from a principle."""
        match = re.search(r"(?:add|include)\s+(.+)", principle, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return "Additional information: [Specify what to add]"

    def _restructure_prompt(self, prompt: str) -> str:
        """Restructure a prompt for better clarity and flow."""
        sentences = re.split(r"[.!?]+", prompt)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) > 1:
            # Find questions and key requests
            questions = [
                s for s in sentences if any(q in s.lower() for q in ["what", "how", "why", "when", "where", "who"])
            ]
            others = [s for s in sentences if s not in questions]

            # Structure: main request, then context
            if questions:
                return f"Main request: {'. '.join(questions)}\n\nContext: {'. '.join(others)}"
            else:
                # Put shortest (likely main point) first
                sentences.sort(key=len)
                return f"Main point: {sentences[0]}\n\nDetails: {'. '.join(sentences[1:])}"

        return prompt

    def _improve_formatting(self, prompt: str) -> str:
        """Improve prompt formatting for better readability."""
        # Add proper capitalization
        sentences = re.split(r"([.!?]+)", prompt)
        formatted = []
        for i, part in enumerate(sentences):
            if part and not re.match(r"[.!?]+", part):
                formatted.append(part[0].upper() + part[1:] if part else part)
            else:
                formatted.append(part)

        return "".join(formatted)

    def _extract_core_request(self, prompt: str) -> str:
        """Extract the core request from a prompt."""
        # Find the main question or instruction
        if "?" in prompt:
            match = re.search(r"([^.]*\?)", prompt)
            if match:
                return match.group(1).strip()

        # Return first sentence as core request
        sentences = re.split(r"[.!?]+", prompt)
        if sentences:
            return sentences[0].strip()

        return prompt
