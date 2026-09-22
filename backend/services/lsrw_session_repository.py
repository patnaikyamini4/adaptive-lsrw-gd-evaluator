from backend.services.mongodb import db


sessions_collection = db["lsrw_sessions"]


# ==================================================
# CREATE SESSION
# ==================================================

def create_session(session):
    """
    Save a new LSRW session to MongoDB.

    A copy is inserted so MongoDB's generated
    ObjectId does not get added to the original
    session dictionary.
    """

    session_to_save = session.copy()

    sessions_collection.insert_one(
        session_to_save
    )

    return session


# ==================================================
# GET SESSION
# ==================================================

def get_session(session_id):
    """
    Retrieve an LSRW session by session_id.

    MongoDB's internal _id is removed before
    returning the document.
    """

    session = sessions_collection.find_one(
        {
            "session_id": session_id
        }
    )

    if session:
        session.pop("_id", None)

    return session


# ==================================================
# UPDATE SESSION
# ==================================================

def update_session(session_id, updates):
    """
    Update an existing LSRW session.

    MongoDB's internal _id is never updated.
    """

    updates_to_save = updates.copy()

    updates_to_save.pop(
        "_id",
        None
    )

    sessions_collection.update_one(
        {
            "session_id": session_id
        },
        {
            "$set": updates_to_save
        }
    )

    return get_session(session_id)