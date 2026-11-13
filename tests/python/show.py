"""Print a solution as plain text: python tests/python/show.py "y' = x*y" 0 1"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "app", "src", "main", "python"))
from deengine import solve

def main():
    args = sys.argv[1:] + ["", "", ""]
    r = json.loads(solve(args[0], args[1], args[2], args[3]))
    if not r["ok"]:
        print("ERROR:", r["error"]); return
    print(r["normal_latex"], "| rhs:", r["rhs_expr"], "|", r["summary"])
    for m in r["methods"]:
        print("==", m["name"], "applicable" if m["applicable"] else "N/A", "--", m["reason"])
        for s in m.get("steps", []):
            print("  *", s["title"], "::", s["text"])
            for l in s["latex"]: print("        ", l)
            for ss in s["substeps"]:
                for l in ss["latex"]: print("           -", ss["title"], l)
        if m["applicable"]: print("  plot:", m["plot"], "verified:", m["verified"])
if __name__ == "__main__":
    main()
