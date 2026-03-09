import sys
import subprocess

def run_agent_review(diff):
    """
    Constructs the review prompt for the Gemini Generalist sub-agent.
    """
    prompt = f"""
    You are a Senior Code Reviewer for the HoopSense project.
    Review the following git diff for:
    1. Compliance with the 'Teach-First' and 'Design-First' mandates in GEMINI.md.
    2. Logic bugs or regressions in the Rust/Python vision logic.
    3. Consistency with the NCAA rule hierarchy.
    4. Completeness of tests for new functionality.

    DIFF:
    {diff}

    Output your findings first, then a summary of 'Approved' or 'Changes Requested'.
    """
    
    print("--- GENERATED AGENT REVIEW PROMPT ---")
    print(prompt)
    print("-------------------------------------")
    return prompt

if __name__ == "__main__":
    try:
        # Check if inside a git repository
        subprocess.check_call(["git", "rev-parse", "--is-inside-work-tree"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Get staged changes
        diff = subprocess.check_output(["git", "diff", "--cached"]).decode("utf-8")
        
        if not diff:
            print("No staged changes to review. Use 'git add' to stage your changes first.")
            sys.exit(0)
        
        run_agent_review(diff)
        
    except subprocess.CalledProcessError:
        print("Error: This script must be run within a git repository.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)
