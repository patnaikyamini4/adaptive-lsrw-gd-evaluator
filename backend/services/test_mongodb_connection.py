from backend.services.mongodb import client, db


try:
    print("Testing MongoDB Atlas connection...")

    result = client.admin.command("ping")

    print("Ping result:", result)
    print("MongoDB connection: SUCCESS")
    print("Database:", db.name)

except Exception as e:
    print("MongoDB connection: FAILED")
    print(type(e).__name__, ":", e)

finally:
    client.close()