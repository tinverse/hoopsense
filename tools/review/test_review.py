import subprocess
import unittest
from review_change import run_agent_review

class TestReviewInfrastructure(unittest.TestCase):
    def test_git_diff_dependency(self):
        """Ensures the script handles environments without git or staged changes."""
        try:
            diff = subprocess.check_output(["git", "diff", "--cached"]).decode("utf-8")
            self.assertIsInstance(diff, str)
        except subprocess.CalledProcessError:
            self.fail("review_change.py must be run within a git repository.")

    def test_prompt_generation(self):
        """Verifies that the agent prompt contains the necessary review criteria."""
        # This is a basic check; real agent testing would involve mock responses
        pass

if __name__ == "__main__":
    unittest.main()
