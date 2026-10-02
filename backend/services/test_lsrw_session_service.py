from datetime import datetime
from backend.services.lsrw_session_service import LSRWSessionService


def test_session_service_lifecycle():
    service = LSRWSessionService()

    # 1. Create session
    session = service.create_session(participant_id="P001")
    assert session["participant_id"] == "P001"
    assert session["status"] == "SCHEDULED"
    assert session["current_module"] is None
    assert session["completed_modules"] == []

    # 2. Start session
    session = service.start_session(session)
    assert session["status"] == "ACTIVE"
    assert session["session_started_at"] is not None

    # 3. Start module
    session = service.start_module(session, module="SPEAKING", question_id="SP001")
    assert session["status"] == "SPEAKING"
    assert session["current_module"] == "SPEAKING"
    assert session["current_question_id"] == "SP001"
    assert session["module_started_at"] is not None

    # 4. Finish module
    session = service.finish_module(session)
    assert session["status"] == "ACTIVE"
    assert session["current_module"] is None
    assert session["current_question_id"] is None
    assert session["module_started_at"] is None
    assert session["completed_modules"] == ["SPEAKING"]

    # 5. Finish session
    session = service.finish_session(session)
    assert session["status"] == "COMPLETED"
    assert session["session_ended_at"] is not None