from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class JWT(db.Model):
    id = db.Column(db.String(255), primary_key=True)
    used = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    expires_at = db.Column(db.DateTime)

    def __init__(self, jti, expires_at):
        self.id = jti
        self.expires_at = expires_at


def verify_jwt(jwt_string, public_key):
    try:
        # Decode the JWT
        header, payload, signature = jose_jwt.decode(jwt_string, public_key, algorithms='RS256',
                                                     options={'verify_signature': True})
        # Check if the jti has been used before
        jwt = JWT.query.filter_by(id=payload["jti"]).first()
        if jwt and jwt.used:
            return "Token has been used before. Please get a new token.", 401
        if jwt and jwt.expires_at < datetime.utcnow():
            return "Token has expired, please get a new token.", 401
        if jwt:
            jwt.used = True
            db.session.commit()
        else:
            new_jwt = JWT(jti=payload["jti"], expires_at=datetime.fromtimestamp(payload["exp"]))
            db.session.add(new_jwt)
            db.session.commit()
        return payload
    except Exception as e:
        return "Invalid token: {}".format(e), 401
