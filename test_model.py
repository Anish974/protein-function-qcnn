"""Quick manual + self-check test for the trained hybrid QNN.

Run:  python test_model.py
      python test_model.py "MKTAYIAKQR..."   # your own sequence
"""
import sys
from src.model import load_model, predict, CLASSES

# Known sample from the notebook -> expected TRANSFERASE
SAMPLE = ("PPYTVVYFPVRGRCAALRMLLADQGQSWKEEVVTVETWQEGSLKASCLYGQLPKFQDGDLTLYQSNTILRHLGR"
          "TLGLYGKDQQEAALVDMVNDGVEDLRCKYISLIYTNYEAGKDDYVKALPGQLKPFETLLSQNQGGKTFIVGDQI"
          "SFADYNLLDLLLIHEVLAPGCLDAFPLLSAYVGRLSARPKLKAFLASPEYVNLPINGNGKQ")

if __name__ == "__main__":
    model = load_model()
    print("Model loaded (q_layers=4).\n")

    seqs = sys.argv[1:] or [SAMPLE]
    for s in seqs:
        cls, conf, _ = predict(model, s)
        print(f"len={len(s):4d}  ->  {cls:30s}  ({conf*100:.1f}%)")

    # self-check: notebook's sample must predict TRANSFERASE
    cls, _, probs = predict(model, SAMPLE)
    assert cls == "TRANSFERASE", f"regression: expected TRANSFERASE, got {cls}"
    assert abs(sum(probs) - 1.0) < 1e-4, "softmax probs must sum to 1"
    print("\nself-check OK: sample -> TRANSFERASE, probs valid.")
