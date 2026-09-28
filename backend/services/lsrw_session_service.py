from datetime import datetime
import uuid


class LSRWSessionService:

    def create_session(
        self,
        participant_id,
        scheduled_start=None
    ):
        session_id = str(uuid.uuid4())

        session = {
            "session_id": session_id,
            "participant_id": participant_id,
            "status": "SCHEDULED",
            "current_module": None,
            "current_question_id": None,
            "scheduled_start": scheduled_start,
            "session_started_at": None,
            "session_ended_at": None,
            "module_started_at": None,
            "completed_modules": [],
            "created_at": datetime.utcnow()
        }

        return session

    def start_session(self, session):
        session["status"] = "ACTIVE"
        session["session_started_at"] = datetime.utcnow()

        return session

    def start_module(
        self,
        session,
        module,
        question_id
    ):
        session["current_module"] = module
        session["current_question_id"] = question_id
        session["module_started_at"] = datetime.utcnow()

        session["status"] = module.upper()

        return session

    def finish_module(self, session):
        module = session["current_module"]

        if module and module not in session["completed_modules"]:
            session["completed_modules"].append(module)

        session["current_module"] = None
        session["current_question_id"] = None
        session["module_started_at"] = None

        session["status"] = "ACTIVE"

        return session

    def finish_session(self, session):
        session["status"] = "COMPLETED"
        session["session_ended_at"] = datetime.utcnow()

        return session