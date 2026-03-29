"""
Classifier Router — Phase 2 + Phase 4 Smart Routing

Two classifier backends:

Phase 2 (TF-IDF):
    TF-IDF vectorizer + multi-label logistic regression.
    Fast, lightweight, uses scikit-learn only. Good baseline.

Phase 4 (Neural):
    Sentence-transformer embeddings + PyTorch MLP.
    Better semantic understanding, shares embedding model with FAISS memory.
    Requires torch + sentence-transformers.

Both classifiers share the same interface (add_sample, train, predict, status)
and return ClassifierPrediction objects so the router doesn't care which is used.

Configurable via classifier.type: "tfidf" | "neural" in config.yaml.
The classifier coexists with rule-based routing:
    - "rule_based" mode: v1 behavior (keywords only)
    - "classifier" mode: classifier only (requires trained model)
    - "hybrid" mode: classifier first, rule-based fallback on low confidence
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from engine.config import ClassifierConfig

logger = logging.getLogger(__name__)


@dataclass
class TrainingSample:
    """A single training example: user prompt → activated modules."""
    prompt: str
    modules: list[str]                  # Which modules were activated
    timestamp: float = 0.0
    feedback: Optional[str] = None      # "good", "bad", or None (unrated)

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "modules": self.modules,
            "timestamp": self.timestamp,
            "feedback": self.feedback,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TrainingSample":
        return cls(
            prompt=data["prompt"],
            modules=data["modules"],
            timestamp=data.get("timestamp", 0.0),
            feedback=data.get("feedback"),
        )


@dataclass
class ClassifierPrediction:
    """Output of the classifier — predicted modules with confidence."""
    predicted_modules: list[str] = field(default_factory=list)
    confidences: dict[str, float] = field(default_factory=dict)
    overall_confidence: float = 0.0
    used_fallback: bool = False
    reasoning: list[str] = field(default_factory=list)


class IntentClassifier:
    """
    TF-IDF + Multi-label Logistic Regression classifier for module routing.

    Lifecycle:
        1. Collect training samples from interactions (automatic)
        2. Train when enough samples accumulate
        3. Predict module activations for new prompts
        4. Fall back to rule-based when confidence is low

    The classifier uses a OneVsRestClassifier with LogisticRegression
    so it can predict multiple modules simultaneously (multi-label).
    """

    def __init__(self, config: ClassifierConfig):
        self.config = config
        self.model_dir = Path(config.model_path)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Training data
        self._training_data: list[TrainingSample] = []
        self._training_file = self.model_dir / "training_data.json"

        # Trained model components
        self._vectorizer = None         # TF-IDF
        self._classifier = None         # LogisticRegression (OneVsRest)
        self._label_classes: list[str] = []  # Known module names

        # State tracking
        self._is_trained = False
        self._samples_since_retrain = 0
        self._train_count = 0

        # Load existing data and model
        self._load_training_data()
        self._load_model()

    # ── Training Data Management ──────────────────────────────

    def _load_training_data(self):
        """Load training samples from disk."""
        if self._training_file.exists():
            try:
                with open(self._training_file, "r") as f:
                    data = json.load(f)
                self._training_data = [TrainingSample.from_dict(s) for s in data]
                logger.info(
                    f"Loaded {len(self._training_data)} training samples "
                    f"for intent classifier"
                )
            except Exception as e:
                logger.error(f"Failed to load training data: {e}")
                self._training_data = []

    def _save_training_data(self):
        """Persist training samples to disk."""
        try:
            with open(self._training_file, "w") as f:
                json.dump(
                    [s.to_dict() for s in self._training_data],
                    f, indent=2,
                )
        except Exception as e:
            logger.error(f"Failed to save training data: {e}")

    def add_sample(self, prompt: str, modules: list[str], feedback: Optional[str] = None):
        """
        Record a training sample from an interaction.
        Called automatically after each routing decision.

        Args:
            prompt: The user's input text
            modules: Which modules were actually activated
            feedback: Optional user feedback ("good"/"bad")
        """
        if not prompt.strip() or not modules:
            return

        sample = TrainingSample(
            prompt=prompt.strip(),
            modules=modules,
            timestamp=time.time(),
            feedback=feedback,
        )
        self._training_data.append(sample)
        self._samples_since_retrain += 1
        self._save_training_data()

        logger.debug(
            f"Added training sample: '{prompt[:50]}...' → {modules} "
            f"(total: {len(self._training_data)}, since retrain: {self._samples_since_retrain})"
        )

        # Auto-retrain if enough new samples
        if (self._samples_since_retrain >= self.config.retrain_interval
                and len(self._training_data) >= self.config.min_training_samples):
            logger.info("Auto-retraining classifier with new samples...")
            self.train()

    def rate_last(self, feedback: str):
        """
        Rate the most recent routing decision.
        Used by /rate command for user feedback.
        """
        if self._training_data:
            self._training_data[-1].feedback = feedback
            self._save_training_data()
            logger.info(f"Rated last routing decision: {feedback}")

    # ── Model Training ────────────────────────────────────────

    def train(self) -> dict:
        """
        Train the classifier on collected samples.

        Returns dict with training stats:
            - samples: number of training samples
            - modules: list of known modules
            - accuracy: cross-validation score (if enough data)
        """
        # Filter out bad-feedback samples (keep good + unrated)
        usable = [s for s in self._training_data if s.feedback != "bad"]

        if len(usable) < self.config.min_training_samples:
            msg = (
                f"Not enough training data: {len(usable)} samples "
                f"(need {self.config.min_training_samples})"
            )
            logger.warning(msg)
            return {"error": msg, "samples": len(usable)}

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.multiclass import OneVsRestClassifier
            from sklearn.preprocessing import MultiLabelBinarizer
            import numpy as np

            # Prepare data
            prompts = [s.prompt for s in usable]
            module_labels = [s.modules for s in usable]

            # Build label space
            mlb = MultiLabelBinarizer()
            y = mlb.fit_transform(module_labels)
            self._label_classes = list(mlb.classes_)

            # TF-IDF features
            self._vectorizer = TfidfVectorizer(
                max_features=self.config.max_features,
                ngram_range=(1, 2),         # Unigrams + bigrams
                stop_words="english",
                sublinear_tf=True,          # Logarithmic TF scaling
                min_df=1,
                max_df=0.95,
            )
            X = self._vectorizer.fit_transform(prompts)

            # Multi-label classifier
            base_clf = LogisticRegression(
                C=1.0,
                max_iter=1000,
                solver="lbfgs",
                class_weight="balanced",    # Handle imbalanced module usage
            )
            self._classifier = OneVsRestClassifier(base_clf)
            self._classifier.fit(X, y)

            self._is_trained = True
            self._samples_since_retrain = 0
            self._train_count += 1

            # Save model
            self._save_model()

            # Calculate basic accuracy (if enough samples for a split)
            accuracy = None
            if len(usable) >= 20:
                from sklearn.model_selection import cross_val_score
                scores = cross_val_score(
                    OneVsRestClassifier(LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")),
                    X, y, cv=min(5, len(usable) // 4), scoring="f1_weighted",
                )
                accuracy = float(np.mean(scores))

            stats = {
                "samples": len(usable),
                "modules": self._label_classes,
                "accuracy": accuracy,
                "train_count": self._train_count,
            }
            logger.info(f"Classifier trained: {stats}")
            return stats

        except ImportError:
            msg = "scikit-learn not installed. Install with: pip install scikit-learn"
            logger.error(msg)
            return {"error": msg}
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {"error": str(e)}

    # ── Prediction ────────────────────────────────────────────

    def predict(self, prompt: str) -> ClassifierPrediction:
        """
        Predict which modules to activate for a given prompt.

        Returns ClassifierPrediction with predicted modules and confidence.
        If the classifier isn't trained or confidence is too low,
        the caller should fall back to rule-based routing.
        """
        prediction = ClassifierPrediction()

        if not self._is_trained or self._classifier is None or self._vectorizer is None:
            prediction.used_fallback = True
            prediction.reasoning.append("Classifier not trained — falling back to rules")
            return prediction

        try:
            import numpy as np

            # Vectorize the prompt
            X = self._vectorizer.transform([prompt])

            # Get probability predictions for each module
            if hasattr(self._classifier, "predict_proba"):
                probas = self._classifier.predict_proba(X)[0]
            else:
                # Some estimators don't have predict_proba — use decision_function
                decisions = self._classifier.decision_function(X)[0]
                # Sigmoid to get pseudo-probabilities
                probas = 1 / (1 + np.exp(-decisions))

            # Map probabilities to module names
            module_probs = {}
            for i, module_name in enumerate(self._label_classes):
                prob = float(probas[i]) if i < len(probas) else 0.0
                module_probs[module_name] = prob

            # Select modules above threshold
            for module_name, prob in sorted(module_probs.items(), key=lambda x: x[1], reverse=True):
                if prob >= self.config.confidence_threshold:
                    prediction.predicted_modules.append(module_name)
                    prediction.confidences[module_name] = round(prob, 3)
                    prediction.reasoning.append(
                        f"[{module_name}] confidence={prob:.3f} (above threshold {self.config.confidence_threshold})"
                    )

            # Overall confidence = max confidence (or 0 if nothing selected)
            if prediction.confidences:
                prediction.overall_confidence = max(prediction.confidences.values())
            else:
                prediction.used_fallback = True
                prediction.reasoning.append(
                    f"No module above confidence threshold ({self.config.confidence_threshold}) — "
                    f"top prediction: {max(module_probs.items(), key=lambda x: x[1]) if module_probs else 'none'}"
                )

            return prediction

        except Exception as e:
            logger.error(f"Classifier prediction failed: {e}")
            prediction.used_fallback = True
            prediction.reasoning.append(f"Prediction error: {e}")
            return prediction

    # ── Model Persistence ─────────────────────────────────────

    def _save_model(self):
        """Save trained model to disk using joblib."""
        try:
            import joblib

            model_file = self.model_dir / "classifier.joblib"
            joblib.dump({
                "vectorizer": self._vectorizer,
                "classifier": self._classifier,
                "label_classes": self._label_classes,
                "train_count": self._train_count,
            }, model_file)
            logger.info(f"Saved classifier model to {model_file}")

        except ImportError:
            logger.warning("joblib not installed — classifier won't persist between sessions")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")

    def _load_model(self):
        """Load a previously trained model from disk."""
        model_file = self.model_dir / "classifier.joblib"
        if not model_file.exists():
            return

        try:
            import joblib

            data = joblib.load(model_file)
            self._vectorizer = data["vectorizer"]
            self._classifier = data["classifier"]
            self._label_classes = data["label_classes"]
            self._train_count = data.get("train_count", 1)
            self._is_trained = True

            logger.info(
                f"Loaded trained classifier (modules: {self._label_classes}, "
                f"train_count: {self._train_count})"
            )

        except ImportError:
            logger.warning("joblib not installed — can't load saved classifier")
        except Exception as e:
            logger.error(f"Failed to load classifier model: {e}")

    # ── Status & Info ─────────────────────────────────────────

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def sample_count(self) -> int:
        return len(self._training_data)

    @property
    def needs_training(self) -> bool:
        """True if we have enough samples but haven't trained yet."""
        return (
            not self._is_trained
            and len(self._training_data) >= self.config.min_training_samples
        )

    def status(self) -> dict:
        """Get classifier status for display."""
        return {
            "trained": self._is_trained,
            "samples": len(self._training_data),
            "min_samples_needed": self.config.min_training_samples,
            "samples_since_retrain": self._samples_since_retrain,
            "known_modules": self._label_classes,
            "train_count": self._train_count,
            "confidence_threshold": self.config.confidence_threshold,
        }


# ══════════════════════════════════════════════════════════════
# Phase 4: Neural Classifier
# ══════════════════════════════════════════════════════════════

class NeuralClassifier:
    """
    Phase 4: Sentence-transformer embeddings + PyTorch MLP classifier.

    Uses the same sentence-transformer model as FAISS memory (all-MiniLM-L6-v2
    by default) so embeddings are semantically rich. A 2-layer MLP maps
    384-dim embeddings → module predictions (multi-label sigmoid output).

    Same interface as IntentClassifier for drop-in replacement.

    Architecture:
        Input (384-dim embedding) → Linear(384, 128) → ReLU → Dropout(0.2)
        → Linear(128, num_modules) → Sigmoid → per-module probabilities
    """

    def __init__(self, config: ClassifierConfig, embedding_model: str = "all-MiniLM-L6-v2"):
        self.config = config
        self.model_dir = Path(config.model_path)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._embedding_model_name = embedding_model

        # Training data (shared format with IntentClassifier)
        self._training_data: list[TrainingSample] = []
        self._training_file = self.model_dir / "training_data.json"

        # Neural model components (lazy-loaded)
        self._embedder = None
        self._mlp = None
        self._label_classes: list[str] = []
        self._embedding_dim: int = 0

        # State
        self._is_trained = False
        self._samples_since_retrain = 0
        self._train_count = 0

        self._load_training_data()
        self._init_embedder()
        self._load_model()

    def _init_embedder(self):
        """Initialize sentence-transformer for embedding prompts (shared instance)."""
        try:
            from engine.embedder import get_embedder
            self._embedder = get_embedder(self._embedding_model_name)
            if self._embedder:
                dummy = self._embedder.encode(["hello"], normalize_embeddings=True)
                self._embedding_dim = dummy.shape[1]
                logger.info(f"Neural classifier embedder: {self._embedding_model_name} (dim={self._embedding_dim})")
            else:
                logger.error("Failed to get shared embedder")
        except ImportError:
            logger.error("sentence-transformers not installed — neural classifier unavailable")

    # ── Training Data Management (same as IntentClassifier) ───

    def _load_training_data(self):
        """Load training samples from disk."""
        if self._training_file.exists():
            try:
                with open(self._training_file, "r") as f:
                    data = json.load(f)
                self._training_data = [TrainingSample.from_dict(s) for s in data]
                logger.info(f"Loaded {len(self._training_data)} training samples for neural classifier")
            except Exception as e:
                logger.error(f"Failed to load training data: {e}")
                self._training_data = []

    def _save_training_data(self):
        """Persist training samples to disk."""
        try:
            with open(self._training_file, "w") as f:
                json.dump([s.to_dict() for s in self._training_data], f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save training data: {e}")

    def add_sample(self, prompt: str, modules: list[str], feedback: Optional[str] = None):
        """Record a training sample from an interaction."""
        if not prompt.strip() or not modules:
            return

        sample = TrainingSample(
            prompt=prompt.strip(),
            modules=modules,
            timestamp=time.time(),
            feedback=feedback,
        )
        self._training_data.append(sample)
        self._samples_since_retrain += 1
        self._save_training_data()

        logger.debug(f"Added neural training sample: '{prompt[:50]}...' → {modules}")

        if (self._samples_since_retrain >= self.config.retrain_interval
                and len(self._training_data) >= self.config.min_training_samples):
            logger.info("Auto-retraining neural classifier...")
            self.train()

    def rate_last(self, feedback: str):
        """Rate the most recent routing decision."""
        if self._training_data:
            self._training_data[-1].feedback = feedback
            self._save_training_data()

    # ── Model Training ────────────────────────────────────────

    def train(self) -> dict:
        """
        Train the MLP classifier on collected samples.

        Embeds all prompts with sentence-transformer, then trains a
        small 2-layer MLP with BCE loss for multi-label classification.
        """
        usable = [s for s in self._training_data if s.feedback != "bad"]

        if len(usable) < self.config.min_training_samples:
            msg = f"Not enough data: {len(usable)} samples (need {self.config.min_training_samples})"
            logger.warning(msg)
            return {"error": msg, "samples": len(usable)}

        if self._embedder is None:
            return {"error": "sentence-transformers not available"}

        try:
            import torch
            import torch.nn as nn
            import numpy as np
            from sklearn.preprocessing import MultiLabelBinarizer

            # Prepare data
            prompts = [s.prompt for s in usable]
            module_labels = [s.modules for s in usable]

            # Build label space
            mlb = MultiLabelBinarizer()
            y = mlb.fit_transform(module_labels)
            self._label_classes = list(mlb.classes_)
            num_modules = len(self._label_classes)

            # Embed all prompts
            X = self._embedder.encode(
                prompts, normalize_embeddings=True, show_progress_bar=False,
            )

            X_tensor = torch.FloatTensor(X)
            y_tensor = torch.FloatTensor(y)

            # Build MLP
            self._mlp = nn.Sequential(
                nn.Linear(self._embedding_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, num_modules),
            )

            # Train
            optimizer = torch.optim.Adam(self._mlp.parameters(), lr=1e-3, weight_decay=1e-4)
            criterion = nn.BCEWithLogitsLoss()

            self._mlp.train()
            epochs = min(200, max(50, len(usable) * 5))

            for epoch in range(epochs):
                optimizer.zero_grad()
                logits = self._mlp(X_tensor)
                loss = criterion(logits, y_tensor)
                loss.backward()
                optimizer.step()

            # Evaluate
            self._mlp.eval()
            with torch.no_grad():
                logits = self._mlp(X_tensor)
                preds = (torch.sigmoid(logits) > self.config.confidence_threshold).float()
                correct = (preds == y_tensor).all(dim=1).sum().item()
                accuracy = correct / len(usable)

            self._is_trained = True
            self._samples_since_retrain = 0
            self._train_count += 1
            self._save_model()

            stats = {
                "samples": len(usable),
                "modules": self._label_classes,
                "accuracy": round(accuracy, 3),
                "train_count": self._train_count,
                "epochs": epochs,
                "final_loss": round(loss.item(), 4),
            }
            logger.info(f"Neural classifier trained: {stats}")
            return stats

        except ImportError:
            return {"error": "torch not installed. Install with: pip install torch"}
        except Exception as e:
            logger.error(f"Neural training failed: {e}")
            return {"error": str(e)}

    # ── Prediction ────────────────────────────────────────────

    def predict(self, prompt: str) -> ClassifierPrediction:
        """Predict module activations using the neural classifier."""
        prediction = ClassifierPrediction()

        if not self._is_trained or self._mlp is None or self._embedder is None:
            prediction.used_fallback = True
            prediction.reasoning.append("Neural classifier not trained — falling back to rules")
            return prediction

        try:
            import torch

            # Embed the prompt
            embedding = self._embedder.encode(
                [prompt], normalize_embeddings=True, show_progress_bar=False,
            )
            x = torch.FloatTensor(embedding)

            # Predict
            self._mlp.eval()
            with torch.no_grad():
                logits = self._mlp(x)
                probs = torch.sigmoid(logits)[0]

            # Map to module names
            module_probs = {}
            for i, module_name in enumerate(self._label_classes):
                module_probs[module_name] = float(probs[i])

            for module_name, prob in sorted(module_probs.items(), key=lambda x: x[1], reverse=True):
                if prob >= self.config.confidence_threshold:
                    prediction.predicted_modules.append(module_name)
                    prediction.confidences[module_name] = round(prob, 3)
                    prediction.reasoning.append(
                        f"[{module_name}] confidence={prob:.3f} (neural, above {self.config.confidence_threshold})"
                    )

            if prediction.confidences:
                prediction.overall_confidence = max(prediction.confidences.values())
            else:
                prediction.used_fallback = True
                top = max(module_probs.items(), key=lambda x: x[1]) if module_probs else ("none", 0)
                prediction.reasoning.append(
                    f"No module above threshold ({self.config.confidence_threshold}) — "
                    f"top: {top[0]}={top[1]:.3f}"
                )

            return prediction

        except Exception as e:
            logger.error(f"Neural prediction failed: {e}")
            prediction.used_fallback = True
            prediction.reasoning.append(f"Neural prediction error: {e}")
            return prediction

    # ── Model Persistence ─────────────────────────────────────

    def _save_model(self):
        """Save the trained MLP to disk."""
        try:
            import torch

            model_file = self.model_dir / "neural_classifier.pt"
            torch.save({
                "mlp_state_dict": self._mlp.state_dict(),
                "mlp_config": {
                    "input_dim": self._embedding_dim,
                    "hidden_dim": 128,
                    "output_dim": len(self._label_classes),
                },
                "label_classes": self._label_classes,
                "train_count": self._train_count,
                "embedding_model": self._embedding_model_name,
            }, model_file)
            logger.info(f"Saved neural classifier to {model_file}")

        except Exception as e:
            logger.error(f"Failed to save neural model: {e}")

    def _load_model(self):
        """Load a previously trained MLP from disk."""
        model_file = self.model_dir / "neural_classifier.pt"
        if not model_file.exists():
            return

        try:
            import torch
            import torch.nn as nn

            data = torch.load(model_file, map_location="cpu", weights_only=False)
            cfg = data["mlp_config"]

            self._mlp = nn.Sequential(
                nn.Linear(cfg["input_dim"], cfg["hidden_dim"]),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(cfg["hidden_dim"], cfg["output_dim"]),
            )
            self._mlp.load_state_dict(data["mlp_state_dict"])
            self._mlp.eval()

            self._label_classes = data["label_classes"]
            self._train_count = data.get("train_count", 1)
            self._is_trained = True

            logger.info(
                f"Loaded neural classifier (modules: {self._label_classes}, "
                f"train_count: {self._train_count})"
            )

        except ImportError:
            logger.warning("torch not installed — can't load neural classifier")
        except Exception as e:
            logger.error(f"Failed to load neural model: {e}")

    # ── Status & Info ─────────────────────────────────────────

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def sample_count(self) -> int:
        return len(self._training_data)

    @property
    def needs_training(self) -> bool:
        return (
            not self._is_trained
            and len(self._training_data) >= self.config.min_training_samples
        )

    def status(self) -> dict:
        return {
            "trained": self._is_trained,
            "type": "neural",
            "samples": len(self._training_data),
            "min_samples_needed": self.config.min_training_samples,
            "samples_since_retrain": self._samples_since_retrain,
            "known_modules": self._label_classes,
            "train_count": self._train_count,
            "confidence_threshold": self.config.confidence_threshold,
            "embedding_model": self._embedding_model_name,
        }


# ══════════════════════════════════════════════════════════════
# Factory
# ══════════════════════════════════════════════════════════════

def create_classifier(config: ClassifierConfig) -> IntentClassifier | NeuralClassifier:
    """
    Factory: create the right classifier based on config.type.

    Falls back to TF-IDF if neural deps (torch, sentence-transformers)
    aren't available.
    """
    classifier_type = config.type

    if classifier_type == "neural":
        try:
            import torch  # noqa: F401
            from engine.embedder import get_embedder  # noqa: F401
            classifier = NeuralClassifier(config, embedding_model=config.embedding_model)
            logger.info("Using neural classifier (sentence-transformers + PyTorch MLP)")
            return classifier
        except ImportError:
            logger.warning(
                "Neural classifier requested but torch or sentence-transformers not installed. "
                "Falling back to TF-IDF classifier."
            )

    classifier = IntentClassifier(config)
    logger.info("Using TF-IDF classifier (scikit-learn)")
    return classifier
