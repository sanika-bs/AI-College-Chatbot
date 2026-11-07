"""Inference engine: preprocess -> saved TF-IDF -> saved classifier -> confidence check -> response mapping."""
import json, os
import joblib, numpy as np
from preprocessing import preprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", 0.35))
FALLBACK = ("I'm not confident I understood that, so I don't want to give you a wrong answer. "
            "Try rephrasing (e.g. 'What are the hostel fees?'), pick a suggested question, or contact the "
            "college office: info@sit-sample.edu / +91-00000-00000 (sample details).")

class Chatbot:
    def __init__(self, threshold=CONFIDENCE_THRESHOLD):
        m = os.path.join(ROOT, "models")
        if not os.path.exists(os.path.join(m, "intent_model.joblib")):   # first run on a fresh deployment
            import train_model; train_model.main()
        self.vectorizer = joblib.load(os.path.join(m, "tfidf_vectorizer.joblib"))   # fitted once, never refitted
        self.model = joblib.load(os.path.join(m, "intent_model.joblib"))
        self.responses = json.load(open(os.path.join(ROOT, "dataset", "responses.json"), encoding="utf-8"))
        self.threshold = threshold

    def reply(self, question):
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be a non-empty string.")
        cleaned = preprocess(question)
        X = self.vectorizer.transform([cleaned])
        known_words = bool(cleaned) and X.getnnz() > 0
        proba = self.model.predict_proba(X)[0]
        order = np.argsort(proba)[::-1]
        best = self.model.classes_[order[0]]; conf = float(proba[order[0]])
        fallback = (not known_words) or conf < self.threshold
        return {"question": question.strip(), "cleaned_question": cleaned,
                "intent": "unknown" if fallback else best, "best_guess": best,
                "confidence": round(conf, 4), "is_fallback": fallback, "threshold": self.threshold,
                "answer": FALLBACK if fallback else self.responses[best],
                "top3": [{"intent": self.model.classes_[i], "confidence": round(float(proba[i]), 4)} for i in order[:3]]}
