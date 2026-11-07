"""Run: python -m unittest tests.test_chatbot -v   (also writes tests/test_results.md from REAL predictions)"""
import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chatbot import Chatbot
from app import app

# (question, expected_intent). 'unknown' = must trigger the low-confidence fallback.
# These questions are NOT in the training dataset.
CASES = [
 ("How do I get a seat in your college?","admission"),("what is the procedure for applying here","admission"),
 ("Which programs can I pursue at your institute?","courses"),("is there a civil engineering department","courses"),
 ("What will I have to pay every year?","fees"),("tell me the tuition for MCA","fees"),
 ("Do I qualify for mba with 55 percent?","eligibility"),("what are the minimum marks needed in 12th for btech","eligibility"),
 ("Can students do internships during the course?","internships"),("how long is the training period in companies","internships"),
 ("What is the average salary offered to students?","placements"),("do top companies recruit from your campus","placements"),
 ("Is there any scholarship for merit students?","scholarships"),("how can poor students get fee waiver","scholarships"),
 ("When do the holidays begin?","academic_calendar"),("what are the semester start dates","academic_calendar"),
 ("What are the passing marks in exams?","examinations"),("where can I see my exam results","examinations"),
 ("Does the college provide hostel rooms?","hostel"),("is food available in the hostel mess","hostel"),
 ("What time does the library close in the evening?","library"),("can I borrow books from the library","library"),
 ("What is the phone number of the college?","contact"),("where is your college located","contact"),
 ("Hello there!","greeting"),("thanks, bye","greeting"),
 # misspelled / incomplete
 ("admision procdure","admission"),("hostle fees","hostel"),("scholarsip details","scholarships"),
 ("placment record","placements"),("libary timings","library"),("exm timetable","examinations"),("fees","fees"),
 # unrelated / unknown -> must fall back
 ("Who won the cricket world cup?","unknown"),("what is the capital of France","unknown"),
 ("tell me a joke about pizza","unknown"),("asdfgh qwerty","unknown"),("how to cook biryani","unknown"),
]
RESULTS = []

class TestChatbot(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.bot = Chatbot(); cls.client = app.test_client()

    def test_intents(self):
        for q, exp in CASES:
            r = self.bot.reply(q); got = r["intent"]
            RESULTS.append((q, exp, got, r["confidence"], "PASS" if got == exp else "FAIL"))
        passed = sum(x[4] == "PASS" for x in RESULTS)
        with open(os.path.join(os.path.dirname(__file__), "test_results.md"), "w") as f:
            f.write(f"# Test results (real predictions)\n\nPassed {passed}/{len(RESULTS)} = {passed/len(RESULTS):.1%}\n\n")
            f.write("| Test question | Expected | Predicted | Confidence | Status |\n|---|---|---|---|---|\n")
            for q, e, g, c, s in RESULTS: f.write(f"| {q} | {e} | {g} | {c:.3f} | {s} |\n")
        print(f"\nIntent tests: {passed}/{len(RESULTS)} passed")
        self.assertGreaterEqual(passed / len(RESULTS), 0.8)

    def test_empty_input(self):
        with self.assertRaises(ValueError): self.bot.reply("   ")
        self.assertEqual(self.client.post("/chat", json={"question": ""}).status_code, 400)
        self.assertEqual(self.client.post("/chat", data="not json").status_code, 400)

    def test_health_and_chat_api(self):
        self.assertEqual(self.client.get("/health").get_json()["status"], "ok")
        self.assertEqual(self.client.get("/").status_code, 200)
        j = self.client.post("/chat", json={"question": "What is the fee structure?"}).get_json()
        self.assertEqual(j["intent"], "fees"); self.assertIn("confidence", j)

if __name__ == "__main__": unittest.main(verbosity=2)
