import certifi
from pymongo import MongoClient

MONGO_URI = "mongodb+srv://cluster0.mcjuw.mongodb.net/?authSource=%24external&authMechanism=MONGODB-X509&retryWrites=true&w=majority&appName=Cluster0"

try:
    client = MongoClient(
        MONGO_URI,
        tls=True,
        tlsCertificateKeyFile=r"X509-cert-2052612786362920307.pem",
        tlsCAFile=certifi.where()
    )

    # Test connection
    print("Connecting to MongoDB Atlas...")

    dbs = client.list_database_names()

    print("✅ Connection Successful!")
    print("Databases:", dbs)

except Exception as e:
    print("❌ Connection Failed")
    print(e)