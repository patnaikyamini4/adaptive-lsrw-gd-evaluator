from backend.services.mongodb import db


responses_collection = db["lsrw_responses"]


def create_response(response):
    """
    Save an LSRW response to MongoDB.
    """

    response_to_save = response.copy()

    responses_collection.insert_one(response_to_save)

    return response


def get_response(session_id, question_id):
    """
    Get a response using session and question IDs.
    """

    response = responses_collection.find_one({
        "session_id": session_id,
        "question_id": question_id
    })

    if response:
        response.pop("_id", None)

    return response


def get_session_responses(session_id):
    """
    Get all responses belonging to a session.
    """

    responses = list(
        responses_collection.find(
            {"session_id": session_id}
        )
    )

    for response in responses:
        response.pop("_id", None)

    return responses