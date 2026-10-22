from cv_assistant.interview import ABOUT_QUESTION, COMPLETED, ROLE_QUESTION, Interview
from cv_assistant.storage import ProfileStore


def make_interview(tmp_path):
    return Interview(ProfileStore(tmp_path / "profiles.sqlite3"))


def test_complete_interview(tmp_path):
    interview = make_interview(tmp_path)
    assert ROLE_QUESTION in interview.handle(1, "/start")
    assert ABOUT_QUESTION in interview.handle(1, "Junior Python Developer")
    assert interview.handle(1, "Учусь Python, сделал Telegram-бота.") == COMPLETED
    profile = interview.store.get(1)
    assert profile.stage == "complete"
    assert profile.target_role == "Junior Python Developer"
    assert profile.about == "Учусь Python, сделал Telegram-бота."


def test_restart_program_keeps_answers_and_next_question(tmp_path):
    make_interview(tmp_path).handle(1, "Python Developer")
    restarted = make_interview(tmp_path)
    assert ABOUT_QUESTION in restarted.handle(1, "/start")
    restarted.handle(1, "Есть учебный проект")
    assert "Python Developer" in restarted.handle(1, "/profile")


def test_users_have_separate_profiles(tmp_path):
    interview = make_interview(tmp_path)
    interview.handle(1, "Python Developer")
    interview.handle(2, "Designer")
    assert interview.store.get(1).target_role == "Python Developer"
    assert interview.store.get(2).target_role == "Designer"
    assert interview.store.get(3).target_role == ""


def test_continue_requires_role_but_allows_missing_about(tmp_path):
    interview = make_interview(tmp_path)
    assert ROLE_QUESTION in interview.handle(1, "/continue")
    interview.handle(1, "Developer")
    assert interview.handle(1, "/continue") == COMPLETED
    assert interview.store.get(1).about == ""


def test_commands_do_not_replace_answers(tmp_path):
    interview = make_interview(tmp_path)
    interview.handle(1, "Developer")
    for command in ["/start@my_bot", "/help", "/unknown", "/profile", "   "]:
        interview.handle(1, command)
    assert interview.store.get(1).stage == "about"
    assert interview.store.get(1).target_role == "Developer"
    assert interview.store.get(1).about == ""


def test_restart_resets_only_current_user(tmp_path):
    interview = make_interview(tmp_path)
    interview.handle(1, "Developer")
    interview.handle(2, "Designer")
    interview.handle(1, "/restart")
    assert interview.store.get(1).stage == "target_role"
    assert interview.store.get(1).target_role == ""
    assert interview.store.get(2).target_role == "Designer"


def test_finished_profile_is_not_overwritten_by_extra_text(tmp_path):
    interview = make_interview(tmp_path)
    interview.handle(1, "Developer")
    interview.handle(1, "Учусь Python")
    interview.handle(1, "Случайное сообщение")
    assert interview.store.get(1).about == "Учусь Python"

