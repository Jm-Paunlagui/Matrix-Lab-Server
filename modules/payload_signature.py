import jwt
from config.configurations import private_key, public_key


def encode_payload(payload):
    """Encode payload with private key"""
    encoded = jwt.encode(payload, private_key, algorithm="RS256")
    return encoded


def decode_payload(encoded):
    """Decode payload with public key"""
    decoded = jwt.decode(encoded, public_key, algorithms=[
                         "RS256"], verify=True)
    return decoded
