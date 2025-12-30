import bcrypt
import getpass

# Get password from user input securely
password_to_hash = getpass.getpass("Enter new password to hash: ").encode('utf-8')

# Generate the hash
hashed_password = bcrypt.hashpw(password_to_hash, bcrypt.gensalt())

print("\nCopy this hashed password to your user database:")
print(hashed_password.decode('utf-8'))