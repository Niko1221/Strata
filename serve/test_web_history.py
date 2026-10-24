"""CPU-only Web Chat history regressions. Node is a development test dependency, not a server dependency."""
import shutil
import subprocess
import unittest
from pathlib import Path


@unittest.skipUnless(shutil.which("node"), "Node.js is required for the Web Chat VM tests")
class WebHistory(unittest.TestCase):
    def run_case(self, case):
        result = subprocess.run(
            [shutil.which("node"), str(Path(__file__).with_suffix(".js")), case],
            capture_output=True, text=True, timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_default_off_preserves_controls_and_request(self):
        self.run_case("default_off")

    def test_edits_and_retries_preserve_model_card_sampling_and_overrides(self):
        self.run_case("history_sampling")

    def test_user_edit_retains_attachments_and_replaces_descendants(self):
        self.run_case("user_edit")

    def test_assistant_edit_removes_generated_metadata_and_tools(self):
        self.run_case("assistant_edit")

    def test_truncate_removes_selected_message_and_descendants(self):
        self.run_case("truncate")

    def test_retry_reuses_previous_user_without_old_answer_or_tool_calls(self):
        self.run_case("retry")

    def test_busy_mutations_block_until_stop_finishes(self):
        self.run_case("stop_then_edit")

    def test_undo_restores_full_history_and_attachments(self):
        self.run_case("undo")

    def test_feature_generation_keeps_undo(self):
        self.run_case("feature_generation_undo")

    def test_new_send_invalidates_old_undo(self):
        self.run_case("new_send_invalidates_undo")

    def test_new_chat_invalidates_old_history_undo(self):
        self.run_case("new_chat_invalidates_undo")

    def test_new_chat_invalidates_undo_after_removing_all_history(self):
        self.run_case("empty_new_chat_invalidates_undo")

    def test_composer_does_not_submit_during_edit(self):
        self.run_case("composer_during_edit")

    def test_empty_reasoning_only_assistant_remains_in_request(self):
        self.run_case("reasoning_only")

    def test_export_contains_current_history_after_edit(self):
        self.run_case("export")

    def test_old_new_chat_undo_cannot_overwrite_after_enabling_feature(self):
        self.run_case("legacy_undo_after_enabling")

    def test_regenerate_does_not_silently_lose_reloaded_attachments(self):
        self.run_case("reloaded_attachments")

    def test_failed_regeneration_preserves_undo(self):
        self.run_case("network_failure")


if __name__ == "__main__":
    unittest.main()
