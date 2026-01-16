"""
Evaluation Module for Financial Insight Generator
==================================================
Production-grade evaluation framework for LLM-generated financial summaries.

Metrics Implemented:
- Faithfulness (NLI-based groundedness)
- Relevance (Query-output semantic alignment)  
- Coherence (Internal consistency via LLM-as-Judge)
- Coverage (Source corpus representation)
- Specificity (Concrete vs. generic content)
- Sentiment Calibration (Alignment between stated sentiment and evidence)
- Temporal Consistency (Inter-run stability)

Author: Emmanuel Adeleye
"""

import os
import json
import logging
import hashlib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import re

from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

from src.llm_module import analyze_with_semantic_search
from src.scraping_module import UnifiedFinancialScraper

# Ensure NLTK resources available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES FOR STRUCTURED RESULTS
# =============================================================================

@dataclass
class FaithfulnessResult:
    """Results from faithfulness evaluation."""
    score: float  # 0-1 scale
    supported_claims: int
    total_claims: int
    claim_details: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass  
class RelevanceResult:
    """Results from relevance evaluation."""
    score: float  # 1-5 scale normalized to 0-1
    raw_score: float  # Original 1-5 scale
    semantic_similarity: float
    explanation: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class CoherenceResult:
    """Results from coherence evaluation."""
    score: float  # 1-5 scale normalized to 0-1
    raw_score: float
    internal_consistency: float
    logical_flow: float
    explanation: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class CoverageResult:
    """Results from coverage evaluation."""
    score: float  # 0-1 scale
    entity_coverage: float
    topic_coverage: float
    key_terms_captured: List[str] = field(default_factory=list)
    missed_terms: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SpecificityResult:
    """Results from specificity evaluation."""
    score: float  # 0-1 scale
    named_entities_count: int
    numeric_claims_count: int
    vague_phrases_count: int
    specificity_ratio: float
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SentimentCalibrationResult:
    """Results from sentiment calibration evaluation."""
    score: float  # 0-1 scale (agreement between stated and detected)
    stated_sentiment: str
    stated_confidence: float
    detected_sentiment: str
    detected_confidence: float
    calibration_error: float
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class TemporalConsistencyResult:
    """Results from temporal consistency evaluation."""
    score: float  # 0-1 scale (mean pairwise similarity)
    run_count: int
    pairwise_similarities: List[float] = field(default_factory=list)
    variance: float = 0.0
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class EvaluationReport:
    """Comprehensive evaluation report."""
    query: str
    timestamp: str
    faithfulness: Optional[FaithfulnessResult] = None
    relevance: Optional[RelevanceResult] = None
    coherence: Optional[CoherenceResult] = None
    coverage: Optional[CoverageResult] = None
    specificity: Optional[SpecificityResult] = None
    sentiment_calibration: Optional[SentimentCalibrationResult] = None
    temporal_consistency: Optional[TemporalConsistencyResult] = None
    aggregate_score: float = 0.0
    
    def to_dict(self) -> Dict:
        result = {
            'query': self.query,
            'timestamp': self.timestamp,
            'aggregate_score': self.aggregate_score
        }
        for field_name in ['faithfulness', 'relevance', 'coherence', 'coverage', 
                          'specificity', 'sentiment_calibration', 'temporal_consistency']:
            field_val = getattr(self, field_name)
            if field_val is not None:
                result[field_name] = field_val.to_dict()
        return result


# =============================================================================
# CORE EVALUATOR CLASS
# =============================================================================

class FinancialInsightEvaluator:
    """
    Production-grade evaluator for LLM-generated financial insights.
    
    Implements industry-standard metrics for evaluating open-ended,
    summary-based LLM outputs without ground truth labels.
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        judge_model: str = "gpt-4o-mini",
        cache_embeddings: bool = True
    ):
        """
        Initialize the evaluator.
        
        Args:
            openai_api_key: OpenAI API key (falls back to env var)
            embedding_model: Sentence transformer model for embeddings
            judge_model: OpenAI model for LLM-as-Judge evaluations
            cache_embeddings: Whether to cache computed embeddings
        """
        self.api_key = openai_api_key or os.getenv("openai_api")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set 'openai_api' env var or pass directly.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.judge_model = judge_model
        
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model)
        
        self.cache_embeddings = cache_embeddings
        self._embedding_cache: Dict[str, np.ndarray] = {}
        
        # Vague phrase patterns for specificity detection
        self.vague_patterns = [
            r'\bsome\b', r'\bmany\b', r'\bvarious\b', r'\bseveral\b',
            r'\bsignificant\b', r'\bsubstantial\b', r'\bconsiderable\b',
            r'\bpotential\b', r'\bpossible\b', r'\bmay\b', r'\bmight\b',
            r'\bcould\b', r'\blikely\b', r'\bperhaps\b', r'\bgenerally\b',
            r'\btypically\b', r'\busually\b', r'\boften\b', r'\bsometimes\b'
        ]
        
        logger.info("FinancialInsightEvaluator initialized successfully")
    
    # -------------------------------------------------------------------------
    # EMBEDDING UTILITIES
    # -------------------------------------------------------------------------
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding with optional caching."""
        cache_key = hashlib.md5(text.encode()).hexdigest()
        
        if self.cache_embeddings and cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
        
        embedding = self.embedder.encode(text, convert_to_numpy=True)
        
        if self.cache_embeddings:
            self._embedding_cache[cache_key] = embedding
        
        return embedding
    
    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts."""
        emb1 = self._get_embedding(text1).reshape(1, -1)
        emb2 = self._get_embedding(text2).reshape(1, -1)
        return float(cosine_similarity(emb1, emb2)[0, 0])
    
    # -------------------------------------------------------------------------
    # LLM-AS-JUDGE UTILITIES
    # -------------------------------------------------------------------------
    
    def _llm_judge(
        self,
        prompt: str,
        system_prompt: str = "You are an expert evaluator of financial text quality."
    ) -> str:
        """Call LLM for evaluation judgments."""
        try:
            response = self.client.chat.completions.create(
                model=self.judge_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM judge call failed: {e}")
            raise
    
    def _parse_score(self, response: str, max_score: int = 5) -> float:
        """Extract numeric score from LLM response."""
        # Look for patterns like "Score: 4" or "4/5" or just "4"
        patterns = [
            r'[Ss]core[:\s]+(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*/\s*\d+',
            r'\b(\d+(?:\.\d+)?)\b'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                score = float(match.group(1))
                return min(score, max_score)
        
        return max_score / 2  # Default to middle if parsing fails
    
    # -------------------------------------------------------------------------
    # CLAIM EXTRACTION
    # -------------------------------------------------------------------------
    
    def _extract_claims(self, output: Dict[str, str]) -> List[str]:
        """Extract individual claims from structured output."""
        claims = []
        
        # Extract from key_insights (numbered list)
        if 'key_insights' in output:
            insights = output['key_insights']
            # Split on numbering patterns
            insight_claims = re.split(r'\d+\.\s*', insights)
            claims.extend([c.strip() for c in insight_claims if c.strip()])
        
        # Extract from key_drivers (comma-separated typically)
        if 'key_drivers' in output:
            drivers = output['key_drivers']
            driver_claims = [d.strip() for d in drivers.split(',')]
            claims.extend([d for d in driver_claims if d])
        
        # Extract from risks
        if 'risks' in output:
            risks = output['risks']
            risk_claims = re.split(r',\s*(?=[A-Z])|;\s*', risks)
            claims.extend([r.strip() for r in risk_claims if r.strip()])
        
        return claims
    
    # -------------------------------------------------------------------------
    # FAITHFULNESS EVALUATION (NLI-based)
    # -------------------------------------------------------------------------
    
    def evaluate_faithfulness(
        self,
        output: Dict[str, str],
        source_chunks: List[str],
        threshold: float = 0.5
    ) -> FaithfulnessResult:
        """
        Evaluate faithfulness using NLI-style entailment checking.
        
        Measures whether claims in the output are supported by source documents.
        Uses semantic similarity as a proxy for entailment when NLI models
        aren't available.
        
        Args:
            output: Generated output dictionary
            source_chunks: List of source text chunks
            threshold: Similarity threshold for claim support
            
        Returns:
            FaithfulnessResult with scores and details
        """
        logger.info("Evaluating faithfulness...")
        
        claims = self._extract_claims(output)
        if not claims:
            logger.warning("No claims extracted from output")
            return FaithfulnessResult(score=0.0, supported_claims=0, total_claims=0)
        
        # Combine source chunks for embedding
        source_text = " ".join(source_chunks)
        source_sentences = sent_tokenize(source_text)
        
        # Pre-compute source embeddings
        source_embeddings = np.array([
            self._get_embedding(sent) for sent in source_sentences
        ])
        
        claim_details = []
        supported_count = 0
        
        for claim in claims:
            claim_embedding = self._get_embedding(claim).reshape(1, -1)
            similarities = cosine_similarity(claim_embedding, source_embeddings)[0]
            
            max_sim = float(np.max(similarities))
            best_match_idx = int(np.argmax(similarities))
            
            is_supported = max_sim >= threshold
            if is_supported:
                supported_count += 1
            
            claim_details.append({
                'claim': claim,
                'supported': is_supported,
                'max_similarity': max_sim,
                'best_evidence': source_sentences[best_match_idx] if source_sentences else ""
            })
        
        score = supported_count / len(claims) if claims else 0.0
        
        logger.info(f"Faithfulness: {supported_count}/{len(claims)} claims supported (score: {score:.3f})")
        
        return FaithfulnessResult(
            score=score,
            supported_claims=supported_count,
            total_claims=len(claims),
            claim_details=claim_details
        )
    
    # -------------------------------------------------------------------------
    # RELEVANCE EVALUATION
    # -------------------------------------------------------------------------
    
    def evaluate_relevance(
        self,
        query: str,
        output: Dict[str, str]
    ) -> RelevanceResult:
        """
        Evaluate relevance of output to the query.
        
        Uses both LLM-as-Judge and semantic similarity for robust assessment.
        
        Args:
            query: Original search query
            output: Generated output dictionary
            
        Returns:
            RelevanceResult with scores and explanation
        """
        logger.info("Evaluating relevance...")
        
        # Flatten output for comparison
        output_text = " ".join(str(v) for v in output.values())
        
        # Compute semantic similarity
        semantic_sim = self._compute_similarity(query, output_text)
        
        # LLM-as-Judge evaluation
        judge_prompt = f"""
                        Evaluate how relevant this financial analysis is to the query.

                        Query: "{query}"

                        Analysis Output:
                        {json.dumps(output, indent=2)}

                        Rate the relevance from 1-5:
                        1 = Completely irrelevant, discusses unrelated topics
                        2 = Mostly irrelevant, only tangentially related
                        3 = Partially relevant, addresses some aspects of the query
                        4 = Mostly relevant, covers main aspects with minor gaps
                        5 = Highly relevant, directly and comprehensively addresses the query

                        Provide your rating as "Score: X" followed by a brief explanation.
                        """
        
        response = self._llm_judge(judge_prompt)
        raw_score = self._parse_score(response)
        normalized_score = (raw_score - 1) / 4  # Normalize to 0-1
        
        # Extract explanation
        explanation = response.split('\n', 1)[-1].strip() if '\n' in response else ""
        
        logger.info(f"Relevance: {raw_score}/5 (semantic sim: {semantic_sim:.3f})")
        
        return RelevanceResult(
            score=normalized_score,
            raw_score=raw_score,
            semantic_similarity=semantic_sim,
            explanation=explanation
        )
    
    # -------------------------------------------------------------------------
    # COHERENCE EVALUATION
    # -------------------------------------------------------------------------
    
    def evaluate_coherence(
        self,
        output: Dict[str, str]
    ) -> CoherenceResult:
        """
        Evaluate internal coherence and logical consistency.
        
        Checks that different parts of the output don't contradict each other
        and that the analysis flows logically.
        
        Args:
            output: Generated output dictionary
            
        Returns:
            CoherenceResult with scores and explanation
        """
        logger.info("Evaluating coherence...")
        
        # Check internal consistency between fields
        # E.g., sentiment should align with risks/insights
        insights = output.get('key_insights', '')
        risks = output.get('risks', '')
        sentiment = output.get('sentiment', '')
        
        # Compute pairwise similarities between sections
        sections = [insights, risks, sentiment]
        sections = [s for s in sections if s]
        
        internal_consistency = 1.0
        if len(sections) >= 2:
            embeddings = np.array([self._get_embedding(s) for s in sections])
            sim_matrix = cosine_similarity(embeddings)
            # Average off-diagonal similarities
            mask = ~np.eye(len(sections), dtype=bool)
            internal_consistency = float(np.mean(sim_matrix[mask]))
        
        # LLM-as-Judge for logical flow
        judge_prompt = f"""
Evaluate the coherence and logical consistency of this financial analysis.

Analysis Output:
{json.dumps(output, indent=2)}

Consider:
1. Do the insights, drivers, and risks tell a consistent story?
2. Does the sentiment align with the stated risks and insights?
3. Is there any contradictory information?
4. Does the analysis flow logically?

Rate coherence from 1-5:
1 = Incoherent, contradictory statements throughout
2 = Poor coherence, significant inconsistencies
3 = Moderate coherence, some minor inconsistencies
4 = Good coherence, mostly consistent with minor issues
5 = Excellent coherence, fully consistent and logical

Provide your rating as "Score: X" followed by a brief explanation.
"""
        
        response = self._llm_judge(judge_prompt)
        raw_score = self._parse_score(response)
        logical_flow = (raw_score - 1) / 4
        
        # Combine metrics
        combined_score = 0.4 * internal_consistency + 0.6 * logical_flow
        
        explanation = response.split('\n', 1)[-1].strip() if '\n' in response else ""
        
        logger.info(f"Coherence: {raw_score}/5 (internal: {internal_consistency:.3f})")
        
        return CoherenceResult(
            score=combined_score,
            raw_score=raw_score,
            internal_consistency=internal_consistency,
            logical_flow=logical_flow,
            explanation=explanation
        )
    
    # -------------------------------------------------------------------------
    # COVERAGE EVALUATION
    # -------------------------------------------------------------------------
    
    def evaluate_coverage(
        self,
        output: Dict[str, str],
        source_chunks: List[str],
        top_n_terms: int = 20
    ) -> CoverageResult:
        """
        Evaluate how well the output covers the source corpus.
        
        Uses TF-IDF to identify key terms in source and checks
        their presence in the output.
        
        Args:
            output: Generated output dictionary
            source_chunks: List of source text chunks
            top_n_terms: Number of top terms to check
            
        Returns:
            CoverageResult with scores and term lists
        """
        logger.info("Evaluating coverage...")
        
        output_text = " ".join(str(v) for v in output.values()).lower()
        source_text = " ".join(source_chunks)
        
        # Extract key terms using TF-IDF
        try:
            vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                ngram_range=(1, 2)
            )
            tfidf_matrix = vectorizer.fit_transform([source_text])
            feature_names = vectorizer.get_feature_names_out()
            
            # Get top terms by TF-IDF score
            scores = tfidf_matrix.toarray()[0]
            top_indices = scores.argsort()[-top_n_terms:][::-1]
            key_terms = [feature_names[i] for i in top_indices]
        except Exception as e:
            logger.warning(f"TF-IDF extraction failed: {e}")
            key_terms = []
        
        # Check which terms appear in output
        captured = []
        missed = []
        
        for term in key_terms:
            if term.lower() in output_text:
                captured.append(term)
            else:
                missed.append(term)
        
        term_coverage = len(captured) / len(key_terms) if key_terms else 0.0
        
        # Entity coverage using semantic similarity
        output_embedding = self._get_embedding(output_text)
        source_embedding = self._get_embedding(source_text)
        entity_coverage = float(cosine_similarity(
            output_embedding.reshape(1, -1),
            source_embedding.reshape(1, -1)
        )[0, 0])
        
        # Combined score
        combined_score = 0.5 * term_coverage + 0.5 * entity_coverage
        
        logger.info(f"Coverage: {len(captured)}/{len(key_terms)} terms (score: {combined_score:.3f})")
        
        return CoverageResult(
            score=combined_score,
            entity_coverage=entity_coverage,
            topic_coverage=term_coverage,
            key_terms_captured=captured,
            missed_terms=missed[:10]  # Limit missed terms in output
        )
    
    # -------------------------------------------------------------------------
    # SPECIFICITY EVALUATION
    # -------------------------------------------------------------------------
    
    def evaluate_specificity(
        self,
        output: Dict[str, str]
    ) -> SpecificityResult:
        """
        Evaluate how specific vs. generic the output is.
        
        Looks for named entities, numeric claims, and penalizes
        vague/hedging language.
        
        Args:
            output: Generated output dictionary
            
        Returns:
            SpecificityResult with counts and scores
        """
        logger.info("Evaluating specificity...")
        
        output_text = " ".join(str(v) for v in output.values())
        
        # Count named entities (simple heuristic: capitalized words not at sentence start)
        words = output_text.split()
        named_entities = sum(1 for i, w in enumerate(words) 
                           if i > 0 and w[0].isupper() and len(w) > 2)
        
        # Count numeric claims
        numeric_pattern = r'\b\d+(?:\.\d+)?(?:%|percent|billion|million|thousand)?\b'
        numeric_claims = len(re.findall(numeric_pattern, output_text))
        
        # Count vague phrases
        vague_count = sum(
            len(re.findall(pattern, output_text, re.IGNORECASE))
            for pattern in self.vague_patterns
        )
        
        # Calculate specificity ratio
        total_words = len(words)
        specific_elements = named_entities + numeric_claims
        vague_elements = vague_count
        
        if total_words > 0:
            specificity_ratio = specific_elements / (specific_elements + vague_elements + 1)
            score = min(1.0, (specific_elements / total_words) * 10)  # Scale appropriately
        else:
            specificity_ratio = 0.0
            score = 0.0
        
        # Adjust score based on vague language
        score = max(0.0, score - (vague_count / total_words) if total_words > 0 else 0)
        
        logger.info(f"Specificity: {named_entities} entities, {numeric_claims} numerics, {vague_count} vague phrases")
        
        return SpecificityResult(
            score=min(1.0, score),
            named_entities_count=named_entities,
            numeric_claims_count=numeric_claims,
            vague_phrases_count=vague_count,
            specificity_ratio=specificity_ratio
        )
    
    # -------------------------------------------------------------------------
    # SENTIMENT CALIBRATION EVALUATION
    # -------------------------------------------------------------------------
    
    def evaluate_sentiment_calibration(
        self,
        output: Dict[str, str]
    ) -> SentimentCalibrationResult:
        """
        Evaluate whether stated sentiment aligns with evidence.
        
        Checks if the sentiment label and confidence percentage
        match what an independent analysis would conclude.
        
        Args:
            output: Generated output dictionary
            
        Returns:
            SentimentCalibrationResult with calibration metrics
        """
        logger.info("Evaluating sentiment calibration...")
        
        sentiment_text = output.get('sentiment', '')
        insights = output.get('key_insights', '')
        risks = output.get('risks', '')
        
        # Parse stated sentiment
        stated_sentiment = 'neutral'
        stated_confidence = 0.5
        
        sentiment_lower = sentiment_text.lower()
        if 'positive' in sentiment_lower:
            stated_sentiment = 'positive'
        elif 'negative' in sentiment_lower:
            stated_sentiment = 'negative'
        
        # Extract percentage if present
        pct_match = re.search(r'(\d+(?:\.\d+)?)\s*%', sentiment_text)
        if pct_match:
            stated_confidence = float(pct_match.group(1)) / 100
        
        # Use LLM to independently assess sentiment
        judge_prompt = f"""
                        Analyze the sentiment of this financial analysis based on the content, ignoring any stated sentiment labels.

                        Key Insights: {insights}
                        Risks: {risks}

                        Based purely on the content:
                        1. What sentiment does this convey? (positive/negative/neutral)
                        2. How confident are you? (0-100%)

                        Respond in format:
                        Sentiment: [positive/negative/neutral]
                        Confidence: [X]%
                        """
        
        response = self._llm_judge(judge_prompt)
        
        # Parse detected sentiment
        detected_sentiment = 'neutral'
        detected_confidence = 0.5
        
        response_lower = response.lower()
        if 'sentiment: positive' in response_lower:
            detected_sentiment = 'positive'
        elif 'sentiment: negative' in response_lower:
            detected_sentiment = 'negative'
        
        conf_match = re.search(r'confidence:\s*(\d+(?:\.\d+)?)', response_lower)
        if conf_match:
            detected_confidence = float(conf_match.group(1)) / 100
        
        # Calculate calibration
        sentiment_match = 1.0 if stated_sentiment == detected_sentiment else 0.0
        confidence_error = abs(stated_confidence - detected_confidence)
        
        # Combined score
        score = 0.6 * sentiment_match + 0.4 * (1 - confidence_error)
        
        logger.info(f"Sentiment calibration: stated={stated_sentiment}, detected={detected_sentiment}")
        
        return SentimentCalibrationResult(
            score=score,
            stated_sentiment=stated_sentiment,
            stated_confidence=stated_confidence,
            detected_sentiment=detected_sentiment,
            detected_confidence=detected_confidence,
            calibration_error=confidence_error
        )
    
    # -------------------------------------------------------------------------
    # TEMPORAL CONSISTENCY EVALUATION
    # -------------------------------------------------------------------------
    
    def evaluate_temporal_consistency(
        self,
        outputs: List[Dict[str, str]]
    ) -> TemporalConsistencyResult:
        """
        Evaluate consistency across multiple runs.
        
        Measures how stable the outputs are when the same query
        is run multiple times.
        
        Args:
            outputs: List of outputs from multiple runs
            
        Returns:
            TemporalConsistencyResult with stability metrics
        """
        logger.info(f"Evaluating temporal consistency across {len(outputs)} runs...")
        
        if len(outputs) < 2:
            logger.warning("Need at least 2 runs for temporal consistency")
            return TemporalConsistencyResult(score=1.0, run_count=len(outputs))
        
        # Convert outputs to text and embed
        output_texts = [" ".join(str(v) for v in o.values()) for o in outputs]
        embeddings = np.array([self._get_embedding(t) for t in output_texts])
        
        # Compute pairwise similarities
        sim_matrix = cosine_similarity(embeddings)
        
        # Extract upper triangle (excluding diagonal)
        pairwise_sims = []
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                pairwise_sims.append(float(sim_matrix[i, j]))
        
        mean_similarity = np.mean(pairwise_sims)
        variance = np.var(pairwise_sims)
        
        logger.info(f"Temporal consistency: mean={mean_similarity:.3f}, var={variance:.4f}")
        
        return TemporalConsistencyResult(
            score=mean_similarity,
            run_count=len(outputs),
            pairwise_similarities=pairwise_sims,
            variance=variance
        )
    
    # -------------------------------------------------------------------------
    # COMPREHENSIVE EVALUATION
    # -------------------------------------------------------------------------
    
    def evaluate(
        self,
        query: str,
        output: Dict[str, str],
        source_chunks: List[str],
        additional_runs: Optional[List[Dict[str, str]]] = None,
        weights: Optional[Dict[str, float]] = None
    ) -> EvaluationReport:
        """
        Run comprehensive evaluation across all metrics.
        
        Args:
            query: Original search query
            output: Generated output dictionary
            source_chunks: List of source text chunks
            additional_runs: Optional list of outputs from repeated runs
            weights: Optional custom weights for aggregate score
            
        Returns:
            EvaluationReport with all metrics
        """
        logger.info("=" * 60)
        logger.info("Starting comprehensive evaluation")
        logger.info("=" * 60)
        
        default_weights = {
            'faithfulness': 0.25,
            'relevance': 0.20,
            'coherence': 0.15,
            'coverage': 0.15,
            'specificity': 0.10,
            'sentiment_calibration': 0.10,
            'temporal_consistency': 0.05
        }
        weights = weights or default_weights
        
        report = EvaluationReport(
            query=query,
            timestamp=datetime.now().isoformat()
        )
        
        # Run all evaluations
        report.faithfulness = self.evaluate_faithfulness(output, source_chunks)
        report.relevance = self.evaluate_relevance(query, output)
        report.coherence = self.evaluate_coherence(output)
        report.coverage = self.evaluate_coverage(output, source_chunks)
        report.specificity = self.evaluate_specificity(output)
        report.sentiment_calibration = self.evaluate_sentiment_calibration(output)
        
        if additional_runs:
            all_runs = [output] + additional_runs
            report.temporal_consistency = self.evaluate_temporal_consistency(all_runs)
        
        # Calculate aggregate score
        scores = {
            'faithfulness': report.faithfulness.score,
            'relevance': report.relevance.score,
            'coherence': report.coherence.score,
            'coverage': report.coverage.score,
            'specificity': report.specificity.score,
            'sentiment_calibration': report.sentiment_calibration.score,
        }
        
        if report.temporal_consistency:
            scores['temporal_consistency'] = report.temporal_consistency.score
        else:
            # Redistribute weight if no temporal consistency
            total_other = sum(v for k, v in weights.items() if k != 'temporal_consistency')
            weights = {k: v/total_other for k, v in weights.items() if k != 'temporal_consistency'}
        
        report.aggregate_score = sum(
            scores[k] * weights.get(k, 0) for k in scores
        )
        
        logger.info("=" * 60)
        logger.info(f"Evaluation complete. Aggregate score: {report.aggregate_score:.3f}")
        logger.info("=" * 60)
        
        return report
    
    # -------------------------------------------------------------------------
    # REPORT GENERATION
    # -------------------------------------------------------------------------
    
    def generate_report_markdown(self, report: EvaluationReport) -> str:
        """Generate a formatted markdown report."""
        
        md = f"""# Financial Insight Evaluation Report

**Query:** {report.query}  
**Timestamp:** {report.timestamp}  
**Aggregate Score:** {report.aggregate_score:.2%}

---

## Summary Metrics

| Metric | Score | Details |
|--------|-------|---------|
| Faithfulness | {report.faithfulness.score:.2%} | {report.faithfulness.supported_claims}/{report.faithfulness.total_claims} claims supported |
| Relevance | {report.relevance.score:.2%} | Raw: {report.relevance.raw_score}/5, Semantic: {report.relevance.semantic_similarity:.2f} |
| Coherence | {report.coherence.score:.2%} | Internal: {report.coherence.internal_consistency:.2f}, Logic: {report.coherence.logical_flow:.2f} |
| Coverage | {report.coverage.score:.2%} | Entity: {report.coverage.entity_coverage:.2f}, Topic: {report.coverage.topic_coverage:.2f} |
| Specificity | {report.specificity.score:.2%} | {report.specificity.named_entities_count} entities, {report.specificity.numeric_claims_count} numerics |
| Sentiment Calibration | {report.sentiment_calibration.score:.2%} | Stated: {report.sentiment_calibration.stated_sentiment}, Detected: {report.sentiment_calibration.detected_sentiment} |
"""
        
        if report.temporal_consistency:
            md += f"| Temporal Consistency | {report.temporal_consistency.score:.2%} | {report.temporal_consistency.run_count} runs, var: {report.temporal_consistency.variance:.4f} |\n"
        
        md += f"""
---

## Detailed Analysis

### Faithfulness Analysis
- **Score:** {report.faithfulness.score:.2%}
- **Supported Claims:** {report.faithfulness.supported_claims} of {report.faithfulness.total_claims}

### Coverage Analysis  
- **Key Terms Captured:** {', '.join(report.coverage.key_terms_captured[:5])}...
- **Notable Gaps:** {', '.join(report.coverage.missed_terms[:5])}...

### Specificity Breakdown
- Named Entities: {report.specificity.named_entities_count}
- Numeric Claims: {report.specificity.numeric_claims_count}
- Vague Phrases: {report.specificity.vague_phrases_count}
- Specificity Ratio: {report.specificity.specificity_ratio:.2f}

---

*Generated by FinancialInsightEvaluator v1.0*
"""
        return md
    
    def save_report(
        self,
        report: EvaluationReport,
        output_dir: str = "evaluation_results",
        formats: List[str] = ["json", "markdown"]
    ) -> Dict[str, str]:
        """
        Save evaluation report to files.
        
        Args:
            report: EvaluationReport to save
            output_dir: Directory for output files
            formats: List of output formats ("json", "markdown")
            
        Returns:
            Dictionary mapping format to filepath
        """
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepaths = {}
        
        if "json" in formats:
            json_path = os.path.join(output_dir, f"eval_report_{timestamp}.json")
            with open(json_path, 'w') as f:
                json.dump(report.to_dict(), f, indent=2)
            filepaths["json"] = json_path
            logger.info(f"Saved JSON report: {json_path}")
        
        if "markdown" in formats:
            md_path = os.path.join(output_dir, f"eval_report_{timestamp}.md")
            with open(md_path, 'w') as f:
                f.write(self.generate_report_markdown(report))
            filepaths["markdown"] = md_path
            logger.info(f"Saved Markdown report: {md_path}")
        
        return filepaths


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_evaluate(
    query: str,
    output: Dict[str, str],
    source_chunks: List[str],
    api_key: Optional[str] = None
) -> Dict[str, float]:
    """
    Quick evaluation returning just the scores.
    
    Args:
        query: Search query
        output: Generated output
        source_chunks: Source text chunks
        api_key: Optional OpenAI API key
        
    Returns:
        Dictionary of metric names to scores
    """
    evaluator = FinancialInsightEvaluator(openai_api_key=api_key)
    report = evaluator.evaluate(query, output, source_chunks)
    
    return {
        'aggregate': report.aggregate_score,
        'faithfulness': report.faithfulness.score,
        'relevance': report.relevance.score,
        'coherence': report.coherence.score,
        'coverage': report.coverage.score,
        'specificity': report.specificity.score,
        'sentiment_calibration': report.sentiment_calibration.score
    }


# =============================================================================
# CLI / MAIN
# =============================================================================

if __name__ == "__main__":
    # Example usage demonstrating the evaluation pipeline
    
    example_output = {
        'key_insights': '1. Global economy expected to grow by 3.2% despite US trade war 2. Small businesses are leading indicators of labor market trends 3. Potential rise in USAR due to geopolitical tensions 4. Speculation on crashing gold prices to reinforce dollar safety 5. Grid capacity shortages being addressed with diesel and gas generators',
        'key_drivers': "Tariffs imposed by Trump on countries doing business with Iran, renewable energy grid upgrades, China's rental of Venezuelan resources, potential digital currency backed by tangible assets",
        'risks': "Potential reduction in employment at mid to large private sector firms, impact on China's economy from US actions, volatility in gold and oil prices due to geopolitical tensions",
        'sentiment': 'Neutral. The corpus contains a mix of positive and negative sentiments, with discussions on potential economic growth and challenges ahead. Overall sentiment leans towards caution and uncertainty. 50% neutral, 30% negative, 20% positive.'
    }
    
    example_sources = [
        "The IMF projects global economic growth of 3.2% for 2024 despite ongoing trade tensions.",
        "Small business hiring has slowed, often a leading indicator for broader employment trends.",
        "Gold prices have shown volatility amid speculation about dollar strength.",
        "Energy grid operators are deploying backup diesel generators to address capacity constraints.",
        "Trade tariffs continue to impact China's manufacturing sector significantly."
    ]
    
    print("=" * 60)
    print("Financial Insight Evaluator - Demo")
    print("=" * 60)
    
    evaluator = FinancialInsightEvaluator()
    report = evaluator.evaluate(
        query="stock market outlook",
        output=example_output,
        source_chunks=example_sources
    )
    
    print("\n" + evaluator.generate_report_markdown(report))
    
    # Save report
    filepaths = evaluator.save_report(report)
    print(f"\nReports saved to: {filepaths}")