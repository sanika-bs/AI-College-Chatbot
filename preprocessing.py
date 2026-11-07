"""Reusable text preprocessing - used identically for training and prediction."""
import re
STOPWORDS = set("""a an the is are am was were be been do does did i me my we our you your to of in on at for and or
can could would will shall should please pls plz tell about there this that it its any some what whats how when where
which who whom with from by as so if also kindly want need like give me us let know""".split())
NORMALIZE = {  # common misspellings / slang -> canonical form
 "admision":"admission","admissoin":"admission","admisssion":"admission","hostle":"hostel","hostal":"hostel","hostl":"hostel",
 "librery":"library","libary":"library","scholarsip":"scholarship","scholarhip":"scholarship","placment":"placement",
 "plcement":"placement","internshp":"internship","intership":"internship","exm":"exam","exms":"exam","examination":"exam",
 "examinations":"exam","eligibilty":"eligibility","calender":"calendar","fess":"fee","colg":"college","clg":"college",
 "u":"you","wat":"what","abt":"about","dept":"department","sem":"semester","hii":"hi","hiii":"hi","hey":"hi","hello":"hi",
 "b.tech":"btech","m.tech":"mtech","applying":"apply","applied":"apply","application":"apply","thx":"thanks","thanx":"thanks",
 "offerd":"offered","eligible":"eligibility","qualification":"eligibility","qualifications":"eligibility","hall":"hall","tution":"tuition","timming":"timing","timmings":"timings",
}
def _stem(tok):
    if len(tok) > 5 and tok.endswith("ies"): return tok[:-3] + "y"      # facilities -> facility
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith(("ss","us","is")): return tok[:-1]  # fees -> fee
    return tok
def tokenize(text):
    text = text.lower().replace("'", "").replace("’", "")
    text = re.sub(r"\b([bm])\.\s?tech\b", r"\1tech", text)       # b.tech -> btech
    return re.findall(r"[a-z0-9]+", text)                         # drops punctuation / special chars / extra spaces
def preprocess(text):
    """lowercase -> strip punctuation -> tokenize -> normalise -> stem -> remove stopwords"""
    if not isinstance(text, str): return ""
    toks = [NORMALIZE.get(t, t) for t in tokenize(text)]
    toks = [_stem(t) for t in toks]
    return " ".join(t for t in toks if t not in STOPWORDS and (len(t) > 1 or t.isdigit()))
