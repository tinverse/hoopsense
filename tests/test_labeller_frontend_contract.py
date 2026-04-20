import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = REPO_ROOT / 'tools' / 'review' / 'labeller' / 'templates' / 'index.html'
APP_JS_PATH = REPO_ROOT / 'tools' / 'review' / 'labeller' / 'static' / 'app.js'


class LabellerFrontendContractTest(unittest.TestCase):
    def test_identity_review_panel_markup_is_present(self):
        template = TEMPLATE_PATH.read_text()
        self.assertIn('id="identity-review-panel"', template)
        self.assertIn('id="identity-review-status"', template)
        self.assertIn('id="identity-review-details"', template)

    def test_identity_review_wiring_is_present_in_app_js(self):
        script = APP_JS_PATH.read_text()
        self.assertIn("const identityReviewPanel = document.getElementById('identity-review-panel');", script)
        self.assertIn('function formatIdentityWorldSummary(frame) {', script)
        self.assertIn('function updateIdentityReviewPanel(frame) {', script)
        self.assertIn('function buildTrackIdentityDeltaLabel(detection) {', script)
        self.assertIn('updateIdentityReviewPanel(frame);', script)

    def test_labeller_app_js_passes_node_syntax_check(self):
        result = subprocess.run(
            ['node', '--check', str(APP_JS_PATH)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)


if __name__ == '__main__':
    unittest.main()
